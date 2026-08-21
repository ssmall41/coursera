#ifndef PLAYER_H
#define PLAYER_H

#include <iostream>
#include "board.h"
#include <memory>

class Player
{
    public: 
        Player(std::shared_ptr<Board> board, bool player_id, int device_id);
        ~Player();
        virtual void make_move() = 0;
    protected:
        std::shared_ptr<Board> game_board;
        bool player_id;
        int device_id;
};

class RandomPlayer : public Player
{
    public:
        RandomPlayer(std::shared_ptr<Board> board, bool player_id, int device_id);
        ~RandomPlayer();
        void make_move() override;
};

#endif // PLAYER_H