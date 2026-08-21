#include "board.h"


// Board class implementation ////////////////////////////////////////

Board::Board(unsigned int width, unsigned int height, unsigned int num_threads) : width(width), height(height), num_threads(num_threads)
{
    std::cout<<"Creating a board..."<<std::endl;

    if(num_threads < height || num_threads < width)
    {
        printf("Error: Not enough threads to check the board!\n");
        exit(1);
    }

    player_boards = new bool[2*width*height];  // Allocate memory for two players, useful for checks

    // Allocate memory for two players on the GPU
    check_cuda_status(cudaMalloc((void**)&d_player_boards, 2*width*height*sizeof(bool)));

    // Default boards to empty
    for(unsigned int i=0;i<2*width*height;i++)
        player_boards[i] = false;
    check_cuda_status(cudaMemcpy(d_player_boards, player_boards, 2*width*height*sizeof(bool), cudaMemcpyHostToDevice));

    int winner = 0;
    check_cuda_status(cudaMalloc((void**)&d_winner, sizeof(int)));
    check_cuda_status(cudaMemcpy(d_winner, &winner, sizeof(int), cudaMemcpyHostToDevice));
}

Board::~Board()
{
    check_cuda_status(cudaFree(d_player_boards));
    check_cuda_status(cudaFree(d_winner));
    delete player_boards;
}

void Board::print_board(bool player)
{
    gpu_print_board<<<1, 1>>>(d_player_boards, width, height, player);
    std::cout<<std::endl;
}


bool Board::set_cell(unsigned int w, bool player)
{
    bool success = false, *d_success;
    check_cuda_status(cudaMalloc((void**)&d_success, sizeof(bool)));
    gpu_set_cell<<<1, num_threads, height*sizeof(bool)>>>(d_player_boards, width, height, w, player, d_success);
    check_cuda_status(cudaGetLastError());
    check_cuda_status(cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
    return success;
}

void Board::print_combined_board()
{
    check_cuda_status(cudaMemcpy(player_boards, d_player_boards, 2*width*height*sizeof(bool), cudaMemcpyDeviceToHost));
    for(int h=height-1;h>=0;h--)
    {
        for(int w=0;w<width;w++)
        {
            if(player_boards[w*height + h])
                printf("0");
            else if(player_boards[width*height + w*height + h])
                printf("1");
            else
                printf(".");
        }
        printf("\n");
    }
    printf("\n");
}

// Check if there's a winner. Returns -1 if no winner, 0 if player 0 wins, 1 if player 1 wins
int Board::get_winner()
{
    int winner;
    dim3 threads(num_threads, num_threads, 1);

    //std::cout<<"Checking if player "<<0<<" has won..."<<std::endl;

    gpu_check_winner<<<1, threads>>>(d_player_boards, width, height, 0, d_winner);
    check_cuda_status(cudaGetLastError());
    check_cuda_status(cudaMemcpy(&winner, d_winner, sizeof(int), cudaMemcpyDeviceToHost));
    if(winner)
        return 0;

    //std::cout<<"Checking if player "<<1<<" has won..."<<std::endl;

    gpu_check_winner<<<1, threads>>>(d_player_boards, width, height, 1, d_winner);
    check_cuda_status(cudaGetLastError());
    check_cuda_status(cudaMemcpy(&winner, d_winner, sizeof(int), cudaMemcpyDeviceToHost));
    if(winner)
        return 1;
    
    return -1;
}

unsigned int Board::get_width()
{
    return width;
}

unsigned int Board::get_height()
{
    return height;
}

bool Board::is_full()
{
    int is_full, *d_is_full;
    dim3 threads(num_threads, num_threads, 1);

    check_cuda_status(cudaMalloc((void**)&d_is_full, sizeof(int)));
    gpu_check_full_board<<<1, threads>>>(d_player_boards, width, height, d_is_full);
    check_cuda_status(cudaGetLastError());
    check_cuda_status(cudaMemcpy(&is_full, d_is_full, sizeof(int), cudaMemcpyDeviceToHost));
    return is_full;
}

// GPU code  //////////////////////////////////////////////////////

__device__ bool gpu_get_cell(bool* player_boards, unsigned int width, unsigned int height, unsigned int w, unsigned int h, bool player)
{
    return player_boards[player*width*height + w*height + h];
}

__global__ void gpu_print_board(bool* player_boards, unsigned int width, unsigned int height, bool player)
{
    int tid = threadIdx.x;

    if(tid == 0)
    {
        for(int h=height-1;h>=0;h--)
        {
            for(int w=0;w<width;w++)
            {
                if(gpu_get_cell(player_boards, width, height, w, h, player))
                    printf("X");
                else
                    printf(".");
            }
            printf("\n");
        }
        printf("\n");
    }
}

// Assumes more threads than height, requires 1 block
__global__ void gpu_set_cell(bool* boards, unsigned int width, unsigned int height, unsigned int w, bool player, bool* success)
{
    int tid = threadIdx.x;
    int h = 0;
    extern __shared__ bool occupied[];  // Requires height bools

    // Check the height of the column to drop a piece
    if(tid < height)
        occupied[tid] = boards[w*height + tid] || boards[width*height + w*height + tid];
    
    // Fill the space with player's piece
    if(tid == 0)
    {
        for(h=0;h<height;h++)
        {
            if(!occupied[h])
            {
                boards[player*width*height + w*height + h] = true;
                break;
            }
        }

        if(h == height)
            *success = false;
        else
            *success = true;
    }
}


// Assumes a 2D, single block of threads, with at least width*height threads
__global__ void gpu_check_winner(bool* player_boards, unsigned int width, unsigned int height, bool player, int* d_winner)
{
    int thread_id_w = threadIdx.x;
    int thread_id_h = threadIdx.y;
    int count = 0;

    if(thread_id_w >= width || thread_id_h >= height)
        return;

    // Quick reset
    if(thread_id_w == 0 && thread_id_h == 0)
    {
        *d_winner = 0;
        //printf("In kernel, checking for player %d...\n", player);
    }
    __syncthreads();

    // Check for horizontal wins
    for(int i=0;i<4;i++)
    {
        if(thread_id_w+i < width)
            count += gpu_get_cell(player_boards, width, height, thread_id_w+i, thread_id_h, player);
    }
    atomicAdd(d_winner, count == 4 ? 1 : 0);  // If a win is found, increment the winner count
    count = 0;

    // Check for vertical wins
    for(int i=0;i<4;i++)
    {
        if(thread_id_h+i < height)
            count += gpu_get_cell(player_boards, width, height, thread_id_w, thread_id_h+i, player);
    }
    atomicAdd(d_winner, count == 4 ? 1 : 0);  // If a win is found, increment the winner count
    count = 0;

    // Check for diagonal wins, top-left to bottom-right
    for(int i=0;i<4;i++)
    {
        if(thread_id_w+i < width && thread_id_h-i < height)
            count += gpu_get_cell(player_boards, width, height, thread_id_w+i, thread_id_h-i, player);
    }
    atomicAdd(d_winner, count == 4 ? 1 : 0);
    count = 0;

    // Check for diagonal wins, bottom-left to top-right
    for(int i=0;i<4;i++)
    {
        if(thread_id_w+i < width && thread_id_h+i < height)
            count += gpu_get_cell(player_boards, width, height, thread_id_w+i, thread_id_h+i, player);
    }
    atomicAdd(d_winner, count == 4 ? 1 : 0);
}

__global__ void gpu_check_full_board(bool* player_boards, unsigned int width, unsigned int height, int* is_full)
{
    int thread_id_w = threadIdx.x;
    int thread_id_h = threadIdx.y;
    bool occupied = true;

    if(thread_id_w >= width || thread_id_h >= height)
        return;

    // Quick reset
    if(thread_id_w == 0 && thread_id_h == 0)
        *is_full = true;
    __syncthreads();

    // Check if the cell is empty
    occupied = gpu_get_cell(player_boards, width, height, thread_id_w, thread_id_h, 0) && 
                gpu_get_cell(player_boards, width, height, thread_id_w, thread_id_h, 1);
    atomicAdd(is_full, occupied ? 1 : 0);
    __syncthreads();

    if(thread_id_w == 0 && thread_id_h == 0)
    {
        if(*is_full == width*height)
            *is_full = true;
        else
            *is_full = false;
    }
}


// Utility functions for CUDA ///////////////////////////////////

void check_cuda_status(cudaError_t status)
{
    if(status)
        printf("There's an error! %s\n", cudaGetErrorString(status));
}
