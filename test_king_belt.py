from hc_board import Board

def test_king_belt():
    board = Board(board_size=640, perimeter=20)
    board.set_initial_position()
    
    king_belt_list = board.king_belt()
    print("King Belt List:")
    print(king_belt_list)

    attack_defend_result = board.attack_defend()
    print("Attack Defend Result:")
    print(attack_defend_result)

if __name__ == "__main__":
    test_king_belt()