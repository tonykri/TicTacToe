from agent import AIPlayer
from game import TicTacToe


agent = AIPlayer(27,9)

g = TicTacToe(agent)
g.trainAI()
g.play()
