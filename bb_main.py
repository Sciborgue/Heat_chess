import pygame, numpy as np
import os
from bb_board import Board
from bb_sidebar2 import Sidebar

class Game:
    def __init__(self, board_size, perimeter, sidebar_width):
        """Initialize the game with board setup and state."""
        self.board = Board(board_size, perimeter)
        self.sidebar = Sidebar(self, width=sidebar_width, height=board_size * 100)
        self.selected_piece = None
        self.move_text = ""
        self.move_log = []
        self.turn = "white"
        self.is_check = False
        self.available_moves = []
        self.predictions = []

        self.indicators = {}
        self.move_annotations = {}  # to store capture, check and mate

        
        # Sidebar parameters (Prediction tuning)
        self.tree_width = 5
        self.tree_depth = 2
        self.method_weights = {
            "heatmap": 1,
            "king_belt": 1,
            "integrity": 1,
            "square_possession": 1,
            "center_control": 1
        }

        self.method_enabled = {
            "heatmap": True,
            "belt": True,
            "integrity": True
        }

    @staticmethod
    def piece_letter(piece_type):
        return {
            "white_pawns": "",
            "white_knights": "N",
            "white_bishops": "B",
            "white_rooks": "R",
            "white_queen": "Q",
            "white_king": "K",
            "black_pawns": "",
            "black_knights": "N",
            "black_bishops": "B",
            "black_rooks": "R",
            "black_queen": "Q",
            "black_king": "K",
        }.get(piece_type, "?")

    @staticmethod
    def square_to_algebraic(square):
        col = square % 8
        row = square // 8
        file = chr(ord('a') + col)
        rank = str(row + 1)
        return file + rank



    def set_sidebar_params(self, tree_width, tree_depth, method_weights, method_enabled):
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.method_weights = method_weights.copy()      # Use .copy() to avoid mutability issues
        self.method_enabled = method_enabled.copy()


    def generate_symbolic_available_moves(self):
        symbolic = []

        if self.selected_piece:
            from_sq = self.selected_piece["square"]
            piece_type = self.selected_piece["type"]
            for to_sq in self.available_moves:
                symbolic.append(self.format_algebraic_move(from_sq, to_sq, piece_type))
        else:
            for from_sq, to_list in self.board.legal_moves.items():
                piece = self.get_piece_by_square(from_sq)
                if not piece:
                    continue
                piece_type = piece["type"]
                for to_sq in to_list:
                    symbolic.append(self.format_algebraic_move(from_sq, to_sq, piece_type))

        return symbolic


    def format_algebraic_move(self, from_sq, to_sq, piece_type):
        def sq(s): return chr(s % 8 + 97) + str(8 - s // 8)

        short_type = piece_type.split("_")[-1]
        piece_letter = {
            "knights": "N", "bishops": "B", "rooks": "R", "queen": "Q", "king": "K"
        }.get(short_type, "")

        # Check if disambiguation needed
        disambiguation = ""
        from_file = from_sq % 8
        for other_from, tos in self.board.legal_moves.items():
            if other_from == from_sq:
                continue
            other = self.board.get_piece(other_from)
            if other and other["type"] == piece_type and to_sq in tos:
                disambiguation = chr(from_file + 97)
                break

        # Annotations
        key = (from_sq, to_sq)
        flags = self.move_annotations.get(key, {})
        capture_symbol = "x" if flags.get("capture") else ""
        suffix = "#" if flags.get("mate") else "+" if flags.get("check") else ""

        return f"{piece_letter}{disambiguation}{capture_symbol}{sq(to_sq)}{suffix}"




    # PREDICTION

    def predict(self, tree_width, tree_depth, method="top_scores"):
        """Selects a move prediction method."""
        if method == "top_scores":
            return self.predict_top_scores(tree_width, tree_depth)
        elif method == "minimax":
            return self.predict_minimax(tree_width, tree_depth)
        else:
            raise ValueError(f"Unknown prediction method: {method}")


    def predict_top_scores(self, tree_width, tree_depth):
        """Ranks moves using influence-based scoring up to a given depth."""
        
        def evaluate_recursive(state, color, depth, last_move=None):
            if depth == 0:
                _, influence = Board.get_available_moves_from_state(state, color, last_move=last_move)
                return sum(
                    bin(bb).count("1")
                    for bitboards in influence.values()
                    for bb in bitboards
                )

            legal_moves = Board.get_legal_moves_from_state(state, color, last_move=last_move)
            scores = []

            for from_square, to_squares in legal_moves.items():
                if not isinstance(from_square, int):
                    continue
                for to_square in to_squares:
                    piece_type = self.get_piece_by_square(from_square)["type"]
                    piece_key = f"{color}_pawns" if "pawns" in piece_type else piece_type
                    move = (piece_key, from_square, to_square)

                    new_state = Board.apply_move_to_state(state, move)
                    next_color = "black" if color == "white" else "white"
                    score = evaluate_recursive(new_state, next_color, depth - 1, last_move=move)
                    scores.append(score)

            return max(scores) if color == "white" else min(scores)

        board_state = self.board.export_state()
        legal_moves = Board.get_legal_moves_from_state(board_state, self.turn, last_move=self.board.last_move if hasattr(self.board, "last_move") else None)
        scored_moves = []

        for from_square, to_squares in legal_moves.items():
            if not isinstance(from_square, int):
                continue
            piece = self.get_piece_by_square(from_square)
            if not piece:
                continue

            for to_square in to_squares:
                move = (piece["type"], from_square, to_square)
                new_state = Board.apply_move_to_state(board_state, move)
                next_color = "black" if self.turn == "white" else "white"
                score = evaluate_recursive(new_state, next_color, tree_depth - 1, last_move=move)
                scored_moves.append((piece, from_square, to_square, score))

        scored_moves.sort(key=lambda x: x[3], reverse=True)
        return scored_moves[:tree_width]

    def predict_minimax(self, tree_width, tree_depth):
        """Minimax search with alpha-beta pruning."""
        # Minimax function (from previous message) can be added here when needed.
        """Minimax algorithm with alpha-beta pruning using heatmap scores."""
        if tree_depth == 0:
            return self.board.calculate_global_heatmap()  # Base case: return evaluation

        possible_moves = self.board.get_available_moves(self.turn)
        
        if maximizing:  # White's turn (maximize)
            max_eval = float("-inf")
            best_moves = []

            for move in possible_moves[:tree_width]:  # Prune by width
                saved_state = self.board.get_bitboard_state()
                self.board.apply_move_bitboard(move)
                self.turn = "black"  # Switch turn

                evaluation = self.predict(tree_width, tree_depth - 1, alpha, beta, maximizing=False)

                self.board.set_bitboard_state(saved_state)  # Undo move
                self.turn = "white"

                if evaluation > max_eval:
                    max_eval = evaluation
                    best_moves = [move]
                elif evaluation == max_eval:
                    best_moves.append(move)

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return best_moves if tree_depth == self.depth else max_eval

        else:  # Black's turn (minimize)
            min_eval = float("inf")

            for move in possible_moves[:tree_width]:
                saved_state = self.board.get_bitboard_state()
                self.board.apply_move_bitboard(move)
                self.turn = "white"

                evaluation = self.predict(tree_width, tree_depth - 1, alpha, beta, maximizing=True)

                self.board.set_bitboard_state(saved_state)  # Undo move
                self.turn = "black"

                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return min_eval

    # INDICATORS
    @staticmethod
    def compute_square_possession(influence_grid):
        white_squares = sum(1 for val in influence_grid if val > 0)
        black_squares = sum(1 for val in influence_grid if val < 0)
        neutral = 64 - white_squares - black_squares
        return white_squares, black_squares, neutral

    # MOVE PIECE

    def get_piece_by_square(self, square):
        """Returns a piece dict at the given square index, or None if empty."""
        mask = 1 << square
        row, col = divmod(square, 8)

        for piece_key, bitboard in self.board.bitboards.items():
            if piece_key.endswith("pieces"):  # Skip aggregate maps
                continue
            if bitboard & mask:
                color = "white" if "white" in piece_key else "black"
                return {
                    "type": piece_key,
                    "color": color,
                    "square": square,
                    "row": row,
                    "col": col
                }
        return None

    def get_square_under_mouse(self, mouse_pos):
        """Determine the board square (row, col) under the mouse position."""
        x, y = mouse_pos
        row = (y-self.board.perimeter) // self.board.square_size
        col = (x-self.board.perimeter) // self.board.square_size
        return row, col

    def select_piece(self, square):
        """Select a piece only if it matches the current player's color and exists at the given position."""
        piece = self.board.get_piece(square)
        if piece and piece['color'] == self.turn:
            self.selected_piece = piece
            self.available_moves = self.board.legal_moves.get(piece["square"], [])
        else:
            # Clear selection if an invalid piece or empty square is clicked
            self.selected_piece = None
            self.available_moves = []
        
        self.sidebar.symbolic_moves = self.generate_symbolic_available_moves()
        
    def make_move(self, piece, to_square, win):
        from_square = piece["square"]
        move = (from_square, to_square)

        from_row, from_col = divmod(from_square, 8)
        to_row, to_col = divmod(to_square, 8)

        # En passant check
        if piece["type"].endswith("pawns") and abs(from_col - to_col) == 1:
            if self.get_piece_by_square(to_square) is None:
                captured_square = from_row * 8 + to_col
                self.board.clear_square(captured_square)

        # Move the piece
        self.board.apply_move_bitboard(move)

        self.move_log.append((self.turn, piece, from_square, to_square, self.indicators))

    # UPDATE

    def update_game(self, tree_width, tree_depth, move):
        self.board.update(move)
        self.predicted_moves = self.predict(tree_width, tree_depth)

        self.turn = "black" if self.turn == "white" else "white"

        # Copy all board indicators at once
        self.indicators = self.board.indicators.copy()

        symbolic_moves = self.generate_symbolic_available_moves()

        self.sidebar.update_sidebar(
            predictions=self.predicted_moves,
            win=None,
            turn=self.turn,
            symbolic_moves=symbolic_moves,
            **self.indicators  
        )



    
    # DRAWING
    
    def draw(self, win):
        self.board.draw(win, self.selected_piece)

        # Use centralized indicators for overlays
        indicators = self.sidebar.indicators

        if self.sidebar.active_score_layer == 1:
            self.draw_overlay_grid(win, indicators.get("heatmap", np.zeros((8, 8))),
                                positive_color=(255, 0, 0), negative_color=(0, 0, 255), label="score")
        elif self.sidebar.active_score_layer == 2:
            self.draw_overlay_grid(win, indicators.get("square_possession", np.zeros((8, 8))),
                                positive_color=(255, 100, 0), negative_color=(0, 100, 255), label="square")
        elif self.sidebar.active_score_layer == 3:
            self.draw_overlay_grid(win, indicators.get("king_threat_grid", np.zeros((8, 8))),
                                positive_color=(200, 0, 200), negative_color=(0, 200, 200), label="belt")
        elif self.sidebar.active_score_layer == 4:
            self.draw_overlay_grid(win, indicators.get("control_center", np.zeros((8, 8))),
                                positive_color=(200, 0, 200), negative_color=(0, 200, 200), label="center")

        self.sidebar.draw(win, self.turn, self.sidebar.predictions)


    def draw_overlay_grid(self, win, grid_data, positive_color=(255, 0, 0), negative_color=(0, 0, 255), label=""):
        """Draws a transparent grid overlay (e.g., heatmap, control, threats)."""
        font = pygame.font.SysFont(None, 20)

        for row in range(8):
            for col in range(8):
                value = grid_data[row, col]
                if value == 0:
                    continue

                if value > 0:
                    alpha = min(180, value * 40)
                    color = (*positive_color[:3], alpha)
                else:
                    alpha = min(180, -value * 40)
                    color = (*negative_color[:3], alpha)

                overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                overlay.fill(color)

                x = col * SQ_SIZE + BOARD_PERIMETER_WIDTH
                y = row * SQ_SIZE + BOARD_PERIMETER_WIDTH  # Flip vertical axis
                win.blit(overlay, (x, y))

                # Optional: Draw influence score as text
                if label:
                    text = font.render(str(value), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2))
                    win.blit(text, text_rect)

"""     def draw_heatmap_layer(self, win):
        for row in range(8):
            for col in range(8):
                value = self.board.heatmap[row][col]
                color = (max(0, value * 10), 0, max(0, -value * 10))
                overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                overlay.fill((*color, 200)) 
                win.blit(overlay, (col * SQ_SIZE, row * SQ_SIZE))

    def draw_square_possession_layer(self, win):
        for row in range(8):
            for col in range(8):
                value = self.board.heatmap[row][col]
                if value > 0:
                    color = (100, 100, 255)
                elif value < 0:
                    color = (255, 100, 100)
                else:
                    color = (80, 80, 80)
                pygame.draw.rect(win, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_king_belt_layer(self, win):
        belt_mask = self.board.get_king_belt_mask(self.turn)
        for square_index in range(64):
            if belt_mask & (1 << square_index):
                row = 7 - (square_index // 8)
                col = square_index % 8
                color = (200, 0, 0) if self.turn == "white" else (0, 0, 200)
                pygame.draw.rect(win, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_center_control_layer(self, win):
        tight_center = {27, 28, 35, 36}
        big_center = {18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45, 50, 51}
        for square_index in range(64):
            row = 7 - (square_index // 8)
            col = square_index % 8
            if square_index in tight_center:
                color = (180, 180, 0)
            elif square_index in big_center:
                color = (80, 80, 0)
            else:
                continue
            pygame.draw.rect(win, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
 """

# Constants for screen size
BOARD_PERIMETER_WIDTH = 20
BOARD_SIZE = 800
SIDEBAR_WIDTH = 900
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH + 2 * BOARD_PERIMETER_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + 2 * BOARD_PERIMETER_WIDTH  # Adjust as needed
SQ_SIZE = (BOARD_SIZE - 2 * BOARD_PERIMETER_WIDTH) // 8
def main():
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Chess Game")
    # Initialize game with specified board size
    game = Game(BOARD_SIZE, BOARD_PERIMETER_WIDTH,SIDEBAR_WIDTH)
    sidebar = Sidebar(game, SIDEBAR_WIDTH, WINDOW_HEIGHT)
    game.sidebar = sidebar
    
    clock = pygame.time.Clock()
    run = True
    FPS = 60
    move_made = False  # Flag to determine when a new move is made
    
    # Have a first update to get the heatmap, draw it etc
    game.update_game(game.sidebar.tree_width, game.sidebar.tree_depth, move=None)
    game.update_game(game.sidebar.tree_width, game.sidebar.tree_depth, move=None)
    while run:
        clock.tick(FPS)  # Limit frame rate

        # Event Handling
        for event in pygame.event.get():
            # QUIT
            if event.type == pygame.QUIT:
                run = False
            # MANUAL MOVES INPUT
            if event.type == pygame.KEYDOWN and sidebar.input_active:
                if event.key == pygame.K_RETURN:
                    sidebar.input_active = False
                    move_str = sidebar.input_text.strip().replace(" ", "")
                    move_successful = False

                    if len(move_str) >= 4 and len(move_str) % 4 == 0:
                        for i in range(0, len(move_str), 4):
                            part = move_str[i:i+4]
                            from_sq = sidebar.algebraic_to_square(part[:2])
                            to_sq = sidebar.algebraic_to_square(part[2:])

                            if from_sq is not None and to_sq is not None:
                                piece = game.board.get_piece_by_square(from_sq)
                                if piece and piece["color"] == game.turn:
                                    legal = game.board.legal_moves.get(from_sq, [])
                                    if to_sq in legal:
                                        game.make_move(piece, to_sq, win)
                                        move_successful = True
                                        game.selected_piece = None
                                        game.piece_selected = False
                                    else:
                                        print("Illegal move:", part)
                                else:
                                    print("Invalid piece or wrong turn at:", part)

                    sidebar.input_text = ""
                    if move_successful:
                        move_made = True
                elif event.key == pygame.K_BACKSPACE:
                    sidebar.input_text = sidebar.input_text[:-1]
                else:
                    sidebar.input_text += event.unicode
            # CLICKING ON THE BOARD
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                row, col = game.get_square_under_mouse(pos)
                clicked_square = 64- ((row+1) * 8) + col
                if game.selected_piece:
                    if clicked_square in game.available_moves:
                        move = game.make_move(game.selected_piece, clicked_square, win)
                        move_made = True
                        game.selected_piece = None
                        game.piece_selected = False
                    else:     
                        game.select_piece(clicked_square)
                        game.piece_selected = bool(game.selected_piece)
                else:
                    #piece = game.get_piece_by_square(clicked_square)
                    game.select_piece(clicked_square)
                    game.piece_selected = bool(game.selected_piece)

            # Handle Sidebar Events
            game.sidebar.handle_event(event)

        # Clear Window
        win.fill((0, 0, 0))

        game.set_sidebar_params(
            game.sidebar.tree_width,
            game.sidebar.tree_depth,
            game.sidebar.method_weights,
            game.sidebar.method_enabled
        )
        # Recalculate heatmap, control, and available moves after a move
        if move_made:
            raw_last_move = game.move_log[-1]
            last_move = (raw_last_move[1]['type'],raw_last_move[2],raw_last_move[3])
            game.update_game(game.sidebar.tree_width, game.sidebar.tree_depth, last_move)
            move_made = False  

        # Draw Game 
        game.draw(win)
        
        pygame.display.update()


if __name__ == "__main__":
    main()
