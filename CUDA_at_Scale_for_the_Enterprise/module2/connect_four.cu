// nvcc -o connect_four *.cu
#include "connect_four.h"


int main()
{
    // Create the board
    unsigned int width = 8, height = 5, num_threads = 16;
    auto b = std::make_shared<Board>(width, height, num_threads);
    srand(time(NULL));

    // Get the devices for each player
    int device_count, p0_device_id, p1_device_id;
    cudaGetDeviceCount(&device_count);
    if(device_count == 0)
    {
        std::cout << "No CUDA devices found." << std::endl;
        return 1;
    }
    if(device_count == 1)
    {
        p0_device_id = 0;
        p1_device_id = 0;
    }
    else
    {
        p0_device_id = 0;
        p1_device_id = 1;
    }

    // Create the players
    RandomPlayer p0(b, 0, p0_device_id);
    RandomPlayer p1(b, 1, p1_device_id);

    // Play the game. Stop when someone wins.
    while(b->get_winner() == -1 && !(b->is_full()))
    {
        p0.make_move();
        if(b->get_winner() != -1 || (b->is_full()))
            break;
        p1.make_move();
        b->print_combined_board();
    }

    // Check who won
    std::cout<<"Game over! Final board:"<<std::endl;
    b->print_combined_board();
    if(b->get_winner() == 0)
        std::cout << "Player 0 wins!" << std::endl;
    else if(b->get_winner() == 1)
        std::cout << "Player 1 wins!" << std::endl;
    else
        std::cout << "It's a draw!" << std::endl;

    return 0;
}
