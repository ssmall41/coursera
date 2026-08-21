#ifndef BOARD_H
#define BOARD_H

#include <iostream>
#include <cuda.h>
#include <vector>

class Board
{
    public:
        Board(unsigned int width, unsigned int height, unsigned int num_threads);
        ~Board();
        void print_board(bool player);
        bool set_cell(unsigned int w, bool player);
        void print_combined_board();
        int get_winner();
        unsigned int get_width();
        unsigned int get_height();
        bool is_full();
        
    private:
        unsigned int width;
        unsigned int height;
        bool *player_boards, *d_player_boards;
        int* d_winner;
        int num_threads;
};


void check_cuda_status(cudaError_t status);

__global__ void gpu_print_board(bool* player_boards, unsigned int width, unsigned int height, bool player);
__device__ bool gpu_get_cell(bool* player_boards, unsigned int width, unsigned int height, unsigned int w, unsigned int h, bool player);
__global__ void gpu_check_winner(bool* player_boards, unsigned int width, unsigned int height, bool player, int* d_winner);
__global__ void gpu_set_cell(bool* boards, unsigned int width, unsigned int height, unsigned int w, bool player, bool* success);
__global__ void gpu_check_full_board(bool* player_boards, unsigned int width, unsigned int height, int* is_full);

#endif // BOARD_H