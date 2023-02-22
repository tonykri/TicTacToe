import numpy as np
from agent import AIPlayer


class TicTacToe:
    def __init__(self):
        self.AIplayer = AIPlayer(27,9)
        self.AItrainer = AIPlayer(27,9)
        self.board = np.array([' '] * 9)
        self.AIwins = 0
        self.AIloses = 0
        self.AIties = 0
    
    def encode_input(self):
        input = list()
        for cell in self.board:
            for ticket in ['X','O',' ']:
                if cell in ticket: input.append(1)
                else: input.append(0)
        return np.array([input])

    def show_board(self):
        print('-------------------')
        print(self.board[0],'|',self.board[1],'|',self.board[2])
        print(self.board[3],'|',self.board[4],'|',self.board[5])
        print(self.board[6],'|',self.board[7],'|',self.board[8])
        print('-------------------')


    def available_moves(self):
        moves = list()
        for idx,cell in enumerate(self.board):
            if cell == ' ': moves.append(idx)
        return moves

    def game_over(self, state, move, next_state):
        board = self.board.reshape(3,3)
        for row in board:
            win = True
            for cel in row:
                if cel in ['O', ' ']: win = False
            if win:
                self.AIplayer.store_episode(state, move, 200, next_state, True)
                self.AItrainer.store_episode(state, move, -200, next_state, True)
                self.AIplayer.train()
                self.AIwins += 1
                return True

        for i in range(3):
            if board[0][i] == board[1][i] == board[2][i] == 'X':
                self.AIplayer.store_episode(state, move, 200, next_state, True)
                self.AItrainer.store_episode(state, move, -200, next_state, True)
                self.AIwins += 1
                self.AIplayer.train()
                return True
        
        if board[0][0] == board[1][1] == board[2][2] == 'X':
            self.AIplayer.store_episode(state, move, 200, next_state, True)
            self.AItrainer.store_episode(state, move, -200, next_state, True)
            self.AIwins += 1
            self.AIplayer.train()
            return True

        if board[0][2] == board[1][1] == board[2][0] == 'X':
            self.AIplayer.store_episode(state, move, 200, next_state, True)
            self.AItrainer.store_episode(state, move, -200, next_state, True)
            self.AIwins += 1
            self.AIplayer.train()
            return True


        for row in board:
            lose = True
            for cel in row:
                if cel in ['X', ' ']: lose = False
            if lose:
                self.AIplayer.store_episode(state, move, -200, next_state, True)
                self.AItrainer.store_episode(state, move, 200, next_state, True)
                self.AIloses += 1
                self.AIplayer.train()
                return True

        for i in range(3):
            if board[0][i] == board[1][i] == board[2][i] == 'O':
                self.AIplayer.store_episode(state, move, -200, next_state, True)
                self.AItrainer.store_episode(state, move, 200, next_state, True)
                self.AIloses += 1
                self.AIplayer.train()
                return True
        
        if board[0][0] == board[1][1] == board[2][2] == 'O':
            self.AIplayer.store_episode(state, move, -200, next_state, True)
            self.AItrainer.store_episode(state, move, 200, next_state, True)
            self.AIloses += 1
            self.AIplayer.train()
            return True

        if board[0][2] == board[1][1] == board[2][0] == 'O':
            self.AIplayer.store_episode(state, move, -200, next_state, True)
            self.AItrainer.store_episode(state, move, 200, next_state, True)
            self.AIloses += 1
            self.AIplayer.train()
            return True
        

        over = True
        for row in board:
            for cel in row:
                if cel in [' ']: over = False
        
        if over:
            self.AIplayer.store_episode(state, move, 50, next_state, True)
            self.AItrainer.store_episode(state, move, 50, next_state, True)
            self.AIties += 1
            self.AIplayer.train()
            return True
        
        self.AIplayer.store_episode(state, move, 0, next_state, False)
        self.AItrainer.store_episode(state, move, 0, next_state, False)
        return False


    def trainAI(self, n_iterations=1000):
        self.show_board()
        first_player = True
        while n_iterations>0:
            if first_player:
                available_moves = self.available_moves()
                state = self.encode_input()
                move = self.AIplayer.compute_action(state,available_moves)
                self.board[move] = 'X'
                next_state = self.encode_input()
                first_player = not first_player
            else:
                available_moves = self.available_moves()
                state = self.encode_input()
                move = self.AItrainer.compute_action(state,available_moves)
                self.board[move] = 'O'
                next_state = self.encode_input()
                first_player = not first_player

            if self.game_over(state, move, next_state):
                self.show_board()
                self.board = np.array([' '] * 9)
                if self.AIplayer.exploration_proba > 0.001:
                    self.AIplayer.update_exploration_probability()
            
            n_iterations -= 1

        print('Games played: ', n_iterations)
        print('AI wins: ', self.AIwins)
        print('AI loses: ', self.AIloses)
        print('Tie: ', self.AIties)


    def play(self):
        first_player = True
        self.board = np.array([' '] * 9)
        while True:
            print('Press number 0 to stop')
            self.show_board()
            state = self.encode_input()
            if first_player:
                move = int(input('Pick an index: ')) - 1
                if move == -1: return
                moves = self.available_moves()
                while move not in moves:
                    print('Give a valid index')
                    move = int(input('Pick an index: ')) - 1
                self.board[move] = 'O'
                first_player = not first_player
            else:
                moves = self.available_moves()
                move = self.AIplayer.compute_action(state, moves)
                self.board[move] = 'X'
                first_player = not first_player
            
            next_state = self.encode_input()
            if self.game_over(state, move, next_state):
                    self.show_board()
                    self.board = np.array([' '] * 9)
