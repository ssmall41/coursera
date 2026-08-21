#include "player.h"


// Random Player class ////////////////////////////////

RandomPlayer::RandomPlayer(std::shared_ptr<Board> board, bool player_id, int device_id) : Player(board, player_id, device_id)
{
    
}

RandomPlayer::~RandomPlayer()
{

}


void RandomPlayer::make_move()
{
    unsigned int width = game_board->get_width();
    unsigned int height = game_board->get_height();
    unsigned int w;
    bool success = false;
    
    cudaSetDevice(device_id);

    while(!success)
    {
        w = rand() % width;
        success = game_board->set_cell(w, player_id);
    }

    check_cuda_status(cudaDeviceSynchronize());
}

// Player class ///////////////////////////////

Player::Player(std::shared_ptr<Board> board, bool player_id, int device_id) : game_board(board), player_id(player_id), device_id(device_id)
{
    
}

Player::~Player()
{

}
