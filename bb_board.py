import pygame
import pandas as pd
import os
import numpy as np

PIECE_SYMBOLS = {
        "white_pawns": "♙",
        "white_knights": "♘",
        "white_bishops": "♗",
        "white_rooks": "♖",
        "white_queen": "♕",
        "white_king": "♔",
        "black_pawns": "♟︎",
        "black_knights": "♞",
        "black_bishops": "♝",
        "black_rooks": "♜",
        "black_queen": "♛",
        "black_king": "♚",
        "white_pieces": "W", 
        "black_pieces": "B",
        "occupied": "O"
    }

def clamp_bitboard(bb):
    return bb & 0xFFFFFFFFFFFFFFFF

def bitboard_to_list(bb):
    out = []
    while bb:
        mask = bb & -bb
        out.append(mask)
        bb ^= mask
    return out

#Visualization for humans
def print_board_from_bitboards(board):
    """Prints a full chessboard from bitboards with symbols. Also prints white_pieces, black_pieces, and occupied."""

    # Main board (pieces only)
    display = [["."] * 8 for _ in range(8)]
    for piece, bb in board.items():
        if piece in ["white_pieces", "black_pieces", "occupied"]:
            continue  # Skip these for the visual board

        symbol = PIECE_SYMBOLS.get(piece, "?")
        temp_bb = bb
        while temp_bb:
            square = (temp_bb & -temp_bb).bit_length() - 1
            rank, file = divmod(square, 8)
            display[7 - rank][file] = symbol
            temp_bb &= temp_bb - 1

    print("\nPiece Positions:")
    for row in display:
        print(" ".join(row))
    print()

    # Summary views of raw aggregates
    for key in ["white_pieces", "black_pieces", "occupied"]:
        if key in board:
            print(f"{key}:")
            view = [["."] * 8 for _ in range(8)]
            bb = board[key]
            symbol = PIECE_SYMBOLS[key]
            while bb:
                square = (bb & -bb).bit_length() - 1
                rank, file = divmod(square, 8)
                view[7 - rank][file] = symbol
                bb &= bb - 1
            for row in view:
                print(" ".join(row))
            print()

""" test_board = {'white_pawns':2257297456238592}
#test_board2 = {10878760}
print_board_from_bitboards(test_board)

ennemy_inf = {'pawn': [65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 34359738368, 137438953472], 'knight': [329728, 10489856], 'bishop': [2560, 2257297456238592], 'rook': [258, 32864], 'queen': [7188], 'king': [8, 32, 2048, 4096, 8192]}

print_board_from_bitboards(ennemy_inf) """
print_board_from_bitboards({'white_bishops':61576676573184})

class Board:
    def __init__(self, board_size, perimeter):
        self.board_size = board_size
        self.square_size = (board_size - 2 * perimeter) // 8  # Calculate the size of each square
        self.perimeter = perimeter 
        self.bitboards = self.init_bitboards()  # Store board state in bitboards
        self.move_annotations = {}  # to store capture, check and mate        
        self.indicators = {
            "heatmap": np.zeros((8, 8), dtype=int),
            "square_possession": np.zeros((8, 8), dtype=int),
            "king_threat_grid": np.zeros((8, 8), dtype=int),
            "control_center": np.zeros((8, 8), dtype=int),

            "white_control": 0,
            "black_control": 0,
            "global_influence_score": 0,

            "white_squares": 0,
            "black_squares": 0,
            "total_squares": 0,

            "white_belt_score": 0,
            "white_threat_count": 0,
            "black_belt_score": 0,
            "black_threat_count": 0,

            "tight_center_score": 0,
            "large_center_score": 0,
        }


        self.fen_log = []  # Stores FEN for debugging & history
        self.predictions = []  # Store evaluated moves
        self.turn = "white"
        
        # Visual board
        self.board = [[None for _ in range(8)] for _ in range(8)]

        
        # Precomputed bitboard values
        self.white_pieces = (self.bitboards["white_pawns"] | self.bitboards["white_knights"] | 
                            self.bitboards["white_bishops"] | self.bitboards["white_rooks"] | 
                            self.bitboards["white_queen"] | self.bitboards["white_king"])

        self.black_pieces = (self.bitboards["black_pawns"] | self.bitboards["black_knights"] | 
                            self.bitboards["black_bishops"] | self.bitboards["black_rooks"] | 
                            self.bitboards["black_queen"] | self.bitboards["black_king"])
        
        self.occupied = self.white_pieces | self.black_pieces
        self.empty = ~self.occupied & 0xFFFFFFFFFFFFFFFF  # Ensure only 64 bits are considered

        # Castling & En Passant
        self.castling_rights = {"white": {"K": True, "Q": True}, "black": {"k": True, "q": True}}
        self.en_passant_target = None  # Square where en passant is possible

        # Updating legal moves
        self.legal_moves = {}
        self.legal_moves = self.get_legal_moves(self.turn,last_move=None)

    def init_bitboards(self):
        """Initialize all bitboards for piece positions and metadata"""
        return {
            "white_pawns": 0x000000000000FF00,
            "black_pawns": 0x00FF000000000000,
            "white_knights": 0x0000000000000042,
            "black_knights": 0x4200000000000000,
            "white_bishops": 0x0000000000000024,
            "black_bishops": 0x2400000000000000,
            "white_rooks": 0x0000000000000081,
            "black_rooks": 0x8100000000000000,
            "white_queen": 0x0000000000000008,
            "black_queen": 0x0800000000000000,
            "white_king": 0x0000000000000010,
            "black_king": 0x1000000000000000,
        }

# GENERAL FUNCTIONS
    def get_bitboard_state(self):
        """Returns a snapshot of the current board state."""
        return (
            self.white_pawns, self.black_pawns, self.white_knights, self.black_knights,
            self.white_bishops, self.black_bishops, self.white_rooks, self.black_rooks,
            self.white_queen, self.black_queen, self.white_king, self.black_king,
            self.black_pieces, self.white_pieces, self.occupied, self.empty
        )

    def set_bitboard_state(self, state):
        """Restores a saved board state."""
        (
            self.white_pawns, self.black_pawns, self.white_knights, self.black_knights,
            self.white_bishops, self.black_bishops, self.white_rooks, self.black_rooks,
            self.white_queen, self.black_queen, self.white_king, self.black_king,
            self.black_pieces, self.white_pieces, self.occupied, self.empty
        ) = state

# MOVES
    def calculate_global_heatmap(self):
        """Computes a score by summing influence of both colors."""
        white_control = self.calculate_control("white")
        black_control = self.calculate_control("black")
        return sum(white_control) - sum(black_control)
    
    
    @staticmethod
    def get_king_belt_mask(king_square):
        """Returns a bitmask of all surrounding squares (belt) around the king."""
        mask = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                row = king_square // 8 + dy
                col = king_square % 8 + dx
                if 0 <= row < 8 and 0 <= col < 8:
                    mask |= 1 << (row * 8 + col)
        return mask

    def compute_king_belt(self, color):
        """Returns (score_sum, threatened_count) in the belt around the king."""
        king_bb = self.bitboards[f"{color}_king"]
        if king_bb == 0:
            return 0, 0

        king_square = king_bb.bit_length() - 1
        belt_mask = self.get_king_belt_mask(king_square)

        score_sum = 0
        threatened_count = 0
        for i in range(64):
            if belt_mask & (1 << i):
                value = self.heatmap[i // 8, i % 8]
                score_sum += value
                if (color == "white" and value < 0) or (color == "black" and value > 0):
                    threatened_count += 1

        return score_sum, threatened_count

    
    def get_legal_moves(self, color, last_move):
        board_state = self.export_state()
        return Board.get_legal_moves_from_state(board_state, color, last_move)

    def export_state(self):
        # Serialize board state
        return {
            "white_pawns": self.bitboards["white_pawns"],
            "black_pawns": self.bitboards["black_pawns"],
            "white_knights": self.bitboards["white_knights"],
            "black_knights": self.bitboards["black_knights"],
            "white_bishops": self.bitboards["white_bishops"],
            "black_bishops": self.bitboards["black_bishops"],
            "white_rooks": self.bitboards["white_rooks"],
            "black_rooks": self.bitboards["black_rooks"],
            "white_queen": self.bitboards["white_queen"],
            "black_queen": self.bitboards["black_queen"],
            "white_king": self.bitboards["white_king"],
            "black_king": self.bitboards["black_king"],
            "white_pieces": self.white_pieces,
            "black_pieces": self.black_pieces,
            "occupied": self.occupied,
            "castling_rights": self.castling_rights.copy()
        }

    @staticmethod
    def move_piece_on_state(state, piece_key, from_square, to_square):
        """Moves a piece and updates occupancy in a given board state dictionary."""
        from_mask = 1 << from_square
        to_mask = 1 << to_square
        captured = False

        # Move the piece
        state[piece_key] ^= from_mask
        state[piece_key] |= to_mask

        # En passant capture
        if "pawns" in piece_key:
            if abs(from_square - to_square) in [7, 9] and not (state["occupied"] & to_mask):
                direction = -8 if "white" in piece_key else 8
                captured_square = to_square + direction
                capture_mask = 1 << captured_square
                opponent_prefix = "black" if "white" in piece_key else "white"

                for k in state:
                    if k.startswith(opponent_prefix) and "pawns" in k:
                        if state[k] & capture_mask:
                            state[k] ^= capture_mask
                            captured = True

        # Castling rook movement
        if "king" in piece_key:
            if from_square == 4 and to_square == 6:
                state["white_rooks"] ^= (1 << 7)
                state["white_rooks"] |= (1 << 5)
            elif from_square == 4 and to_square == 2:
                state["white_rooks"] ^= (1 << 0)
                state["white_rooks"] |= (1 << 3)
            elif from_square == 60 and to_square == 62:
                state["black_rooks"] ^= (1 << 63)
                state["black_rooks"] |= (1 << 61)
            elif from_square == 60 and to_square == 58:
                state["black_rooks"] ^= (1 << 56)
                state["black_rooks"] |= (1 << 59)

        # Standard capture
        opponent_prefix = "black" if "white" in piece_key else "white"
        for key in state:
            if key.startswith(opponent_prefix) and (state[key] & to_mask):
                state[key] ^= to_mask
                captured = True

        # Update metadata
        state["white_pieces"] = sum(state[k] for k in state if k.startswith("white") and not k.endswith("pieces"))
        state["black_pieces"] = sum(state[k] for k in state if k.startswith("black") and not k.endswith("pieces"))
        state["occupied"] = state["white_pieces"] | state["black_pieces"]

        return state

    @staticmethod
    def is_king_threatened(state, color, last_move=None):
        """Check if the king of the given color is under threat in the given state."""
        king_bb = state[f"{color}_king"]
        if king_bb == 0:
            return False
        king_square = king_bb.bit_length() - 1
        enemy_color = "black" if color == "white" else "white"
        _, influence = Board.get_available_moves_from_state(state, enemy_color, last_move=last_move)
        all_influence = 0
        for bb_list in influence.values():
            for bb in bb_list:
                all_influence |= bb
        return bool(all_influence & (1 << king_square))

    @staticmethod
    def apply_move_to_state(board_state, move):
        """Simulates a move on a copied state and returns the new state."""
        piece_key, from_sq, to_sq = move
        new_state = board_state.copy()
        return Board.move_piece_on_state(new_state, piece_key, from_sq, to_sq)

    def apply_move_bitboard(self, move):
        print(self.bitboards)
        print_board_from_bitboards(self.bitboards)
        
        """Applies a move using bitboard manipulation (modifies internal state)."""
        start_square, end_square = move
        start_mask = 1 << start_square

        piece_key = None
        for key in self.bitboards:
            if key in ["white_pieces", "black_pieces", "occupied"]:
                continue
            if self.bitboards[key] & start_mask:
                piece_key = key
                break

        if piece_key is None:
            raise ValueError(f"No piece found at square {start_square}.")

        # Use shared logic on self.bitboards directly
        Board.move_piece_on_state(self.bitboards, piece_key, start_square, end_square)

        # Revoke castling rights if necessary
        if piece_key == "white_king":
            self.castling_rights["K"] = False
            self.castling_rights["Q"] = False
        elif piece_key == "black_king":
            self.castling_rights["k"] = False
            self.castling_rights["q"] = False
        elif piece_key == "white_rooks":
            if start_square == 0:
                self.castling_rights["Q"] = False
            elif start_square == 7:
                self.castling_rights["K"] = False
        elif piece_key == "black_rooks":
            if start_square == 56:
                self.castling_rights["q"] = False
            elif start_square == 63:
                self.castling_rights["k"] = False

        print("Before move:")
        print(self.bitboards)
        print_board_from_bitboards(self.bitboards)
        self.update_board_state()
        print("After move:")
        print(self.bitboards)
        print_board_from_bitboards(self.bitboards)

    def update_occupancy(self):
        """Recomputes overall occupancy and side-specific occupancy."""
        self.occupied = 0
        self.bitboards["white_pieces"] = 0
        self.bitboards["black_pieces"] = 0

        for key, bb in self.bitboards.items():
            if key.endswith("pieces"):
                continue
            self.occupied |= bb
            if key.startswith("white"):
                self.bitboards["white_pieces"] |= bb
            elif key.startswith("black"):
                self.bitboards["black_pieces"] |= bb

#Drawing 
    def draw(self, win, selected_piece):
        """Draws the chessboard and pieces."""
        colors = [(232, 235, 239), (125, 135, 150)]  # White and black colors for the squares
        # Draw the board
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(win, color, (col * self.square_size + self.perimeter, row * self.square_size + self.perimeter, self.square_size, self.square_size))

        font = pygame.font.SysFont(None, 20)
        for i in range(8):
            # Files A–H
            file_label = font.render(chr(65 + i), True, (200, 200, 200))
            win.blit(file_label, (self.perimeter + i * self.square_size + self.square_size // 2 - 6, 8 * self.square_size + self.perimeter + 5))

            # Ranks 1–8 (from bottom up)
            rank_label = font.render(str(8 - i), True, (200, 200, 200))
            win.blit(rank_label, (self.perimeter - 15, self.perimeter + i * self.square_size + self.square_size // 2 - 8))

        # Draw pieces (assuming we have a function to get piece positions)
        for piece, bitboard in self.bitboards.items():
            self.draw_pieces(win, bitboard, piece)
            
        # Highlight the selected piece
        if selected_piece:
            x = selected_piece['col'] * self.square_size + self.perimeter
            y = (7-selected_piece['row']) * self.square_size + self.perimeter
            pygame.draw.rect(win, (0, 0, 0), (x, y, self.square_size, self.square_size), 5)  # Thick black border
        
    def draw_pieces(self, win, bitboard, piece):
        """Draws a piece on the board using bitboard positions."""
        piece_images = {
            "white_pawns": pygame.image.load("assets/wp.png"),
            "black_pawns": pygame.image.load("assets/bp.png"),
            "white_knights": pygame.image.load("assets/wn.png"),
            "black_knights": pygame.image.load("assets/bn.png"),
            "white_bishops": pygame.image.load("assets/wb.png"),
            "black_bishops": pygame.image.load("assets/bb.png"),
            "white_rooks": pygame.image.load("assets/wr.png"),
            "black_rooks": pygame.image.load("assets/br.png"),
            "white_queen": pygame.image.load("assets/wq.png"),
            "black_queen": pygame.image.load("assets/bq.png"),
            "white_king": pygame.image.load("assets/wk.png"),
            "black_king": pygame.image.load("assets/bk.png"),
        }
        
        if piece in piece_images:
            img = pygame.transform.scale(piece_images[piece], (self.square_size, self.square_size))

            while bitboard:
                square = (bitboard & -bitboard).bit_length() - 1
                row, col = divmod(square, 8)
                win.blit(img, (col * self.square_size + self.perimeter, (7 - row) * self.square_size + self.perimeter))
                bitboard &= bitboard - 1  # Remove the LSB (processed piece)

#Board Update
    def update(self, move):
        """Handles a move and updates available moves."""
        # Switch turn
        self.turn = "black" if self.turn == "white" else "white"
        # Recompute legal moves for the new player
        self.legal_moves = self.get_legal_moves(self.turn, last_move=move)
        self.move_annotations = {}

        # For each legal move, precompute flags:
        for from_sq, to_sqs in self.legal_moves.items():
            piece = self.get_piece_by_square(from_sq)
            if not piece:
                continue
            piece_type = piece["type"]

            for to_sq in to_sqs:
                is_capture = self.is_capture(from_sq, to_sq)
                move = (piece_type, from_sq, to_sq)
                next_state = Board.apply_move_to_state(self.export_state(), move)
                is_check = Board.is_king_threatened(next_state, self.turn)
                legal_replies = Board.get_legal_moves_from_state(next_state, self.turn)
                is_mate = is_check and not any(legal_replies.values())

                self.move_annotations[(from_sq, to_sq)] = {
                    "capture": is_capture,
                    "check": is_check,
                    "mate": is_mate
                }

        self.update_heatmaps(move)
        
    def update_board_state(self):
        """Recomputes all dynamic board data: occupancy, piece aggregates, and empty squares."""
        white_total = 0
        black_total = 0
        self.occupied = 0

        for key in self.bitboards:
            if key.endswith("pieces"):
                continue  # We'll compute these below

            bb = self.bitboards[key]
            self.occupied |= bb

            if key.startswith("white"):
                white_total |= bb
            elif key.startswith("black"):
                black_total |= bb

        # Update bitboards for aggregate sides
        self.bitboards["white_pieces"] = white_total
        self.bitboards["black_pieces"] = black_total

        # Store also as top-level attributes (for convenience)
        self.white_pieces = white_total
        self.black_pieces = black_total
        self.empty = ~self.occupied & 0xFFFFFFFFFFFFFFFF
    
    def update_heatmaps(self, last_move):
        self.indicators["heatmap"] = self.compute_influence_heatmap(last_move)
        hm = self.indicators["heatmap"]
        self.indicators["white_control"] = int((hm > 0).sum())
        self.indicators["black_control"] = int((hm < 0).sum())
        self.indicators["global_influence_score"] = int(hm.sum())

        self.indicators["square_possession"],self.indicators["white_squares"], self.indicators["black_squares"], self.indicators["total_squares"]  = Board.calculate_square_possession(hm)

        self.compute_king_threat_grid()
        self.compute_control_center()

    
    def compute_influence_heatmap(self, last_move):
        """Compute influence heatmap using vectorized NumPy operations."""
        heatmap = np.zeros((8, 8), dtype=int)

        for color, sign in [("white", 1), ("black", -1)]:
            _, influence = Board.get_available_moves_from_state(self.export_state(), color, last_move)
            
            for bitboards in influence.values():
                # Convert each bitboard to an array of square indices
                for bb in bitboards:
                    indices = Board.bitboard_to_squares(bb)  # e.g. [8, 16, 23]
                    if indices:
                        rows = 7 - np.array(indices) // 8  # Flip for display alignment
                        cols = np.array(indices) % 8
                        heatmap[rows, cols] += sign

        self.heatmap = heatmap
        return heatmap

    @staticmethod
    def calculate_square_possession(heatmap):
        # Returns: grid, white_squares, black_squares, total_squares
        white_mask = heatmap > 0
        black_mask = heatmap < 0

        grid = np.zeros((8, 8), dtype=int)
        grid[white_mask] = 1
        grid[black_mask] = -1

        w = np.count_nonzero(white_mask)
        b = np.count_nonzero(black_mask)
        t = 64 - (w + b)

        return grid, w, b, t

    def compute_king_threat_grid(self):
        """Compute threat grid around both kings using current heatmap."""
        grid = np.zeros_like(self.indicators["heatmap"])
        heatmap = self.indicators["heatmap"]

        for color in ["white", "black"]:
            king_bb = self.bitboards.get(f"{color}_king", 0)
            if king_bb == 0:
                continue

            king_index = king_bb.bit_length() - 1
            row, col = divmod(king_index, 8)
            row = 7 - row  # Flip vertically (rank to row)

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < 8 and 0 <= c < 8:
                        grid[r, c] = heatmap[r, c]

        self.indicators["king_threat_grid"] = grid

    def compute_control_center(self):
        """Calculates control values and influence grid for center regions."""
        heatmap = self.indicators["heatmap"]

        small_center = [(3, 3), (3, 4), (4, 3), (4, 4)]
        big_center = [
            (2, 2), (2, 3), (2, 4), (2, 5),
            (3, 2), (3, 3), (3, 4), (3, 5),
            (4, 2), (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4)
        ]

        control_grid = np.zeros_like(heatmap)

        for r, c in big_center:
            control_grid[r, c] = heatmap[r, c]  # Fill for display layer

        small_score = sum(heatmap[r, c] for r, c in small_center)
        large_score = sum(heatmap[r, c] for r, c in big_center)

        self.indicators["control_center"] = control_grid
        self.indicators["tight_center_score"] = int(small_score)
        self.indicators["large_center_score"] = int(large_score)


#Pieces moves & influence
    def get_piece(self, square):
        """Return a piece dict from a square index (0–63), or None if empty."""
        mask = 1 << square
        row, col = divmod(square, 8)
        for piece, bitboard in self.bitboards.items():
            if bitboard & mask:
                color = "white" if "white" in piece else "black"
                return {
                    "type": piece,
                    "color": color,
                    "square": square,
                    "row": row,
                    "col": col
                }
        return None

    def clear_square(self, square):
        """Removes any piece from the given square and updates bitboards."""
        row, col = divmod(square, 8)

        piece = self.board[row][col]
        if piece:
            piece_type = piece["type"]

            # Clear from visual board
            self.board[row][col] = None

            # Clear from bitboard
            self.bitboards[piece_type] &= ~(1 << square)

            # Recompute occupancy
            self.update_board_state()

    @staticmethod
    def get_legal_moves_from_state(board_state, color, last_move=None):
        """Returns only legal moves (not pseudo), keyed as from → [to_squares]."""
        legal_moves = {}
        annotations = {}  # Track move metadata like check/mate

        enemy_color = "black" if color == "white" else "white"

        # Get pseudo-legal moves + capture annotations
        available_moves, _, pseudo_annotations = Board.get_available_moves_from_state(board_state, color, last_move)

        for from_sq, to_list in available_moves.items():
            for to_sq in to_list:
                piece_key = Board.get_piece_key_at_square(from_sq, board_state, color)
                move = (piece_key, from_sq, to_sq)
                new_state = Board.apply_move_to_state(board_state, move)

                # Recompute king position (in new_state)
                king_bb = new_state[f"{color}_king"]
                king_pos = king_bb.bit_length() - 1 if king_bb else -1

                # Compute enemy influence
                _, enemy_influence, _ = Board.get_available_moves_from_state(new_state, enemy_color, last_move=move)
                all_threats = 0
                for group in enemy_influence.values():
                    for bb in group:
                        all_threats |= bb

                # If king is not threatened, the move is legal
                if king_pos != -1 and not (all_threats & (1 << king_pos)):
                    legal_moves.setdefault(from_sq, []).append(to_sq)

                    # Start from existing capture annotation
                    annotation = pseudo_annotations.get((from_sq, to_sq), {}).copy()
                    annotation["check"] = False
                    annotation["mate"] = False

                    # Now check if this move puts enemy king in check
                    enemy_king_bb = new_state[f"{enemy_color}_king"]
                    enemy_king_sq = enemy_king_bb.bit_length() - 1 if enemy_king_bb else -1
                    if enemy_king_sq != -1 and (all_threats & (1 << enemy_king_sq)):
                        annotation["check"] = True

                        # Check if this is mate (no legal replies)
                        enemy_legal,_ = Board.get_legal_moves_from_state(new_state, enemy_color, last_move=move)
                        if not any(enemy_legal.values()):
                            annotation["mate"] = True

                    annotations[(from_sq, to_sq)] = annotation

        return legal_moves, annotations


    @staticmethod
    def get_legal_moves_from_state2(board_state, color, last_move=None):
        """Returns only legal moves (not pseudo), keyed as from → [to_squares]."""
        legal_moves = {}
        enemy_color = "black" if color == "white" else "white"

        # Get all pseudo-legal moves
        available_moves, _ = Board.get_available_moves_from_state(board_state, color, last_move)

        for from_sq, to_list in available_moves.items():
            for to_sq in to_list:
                piece_key = Board.get_piece_key_at_square(from_sq, board_state, color)
                move = (piece_key, from_sq, to_sq)
                new_state = Board.apply_move_to_state(board_state, move)

                # Recompute king position (in new_state)
                king_bb = new_state[f"{color}_king"]
                king_pos = king_bb.bit_length() - 1 if king_bb else -1

                # Check if king is in check
                _, enemy_influence = Board.get_available_moves_from_state(new_state, enemy_color)
                all_threats = 0
                for group in enemy_influence.values():
                    for bb in group:
                        all_threats |= bb

                if king_pos != -1 and not (all_threats & (1 << king_pos)):
                    legal_moves.setdefault(from_sq, []).append(to_sq)

        return legal_moves

    @staticmethod
    def get_available_moves_from_state(board_state, color, last_move=None):
        """Return all available pseudo-legal moves and influence for the given color."""
        moves = {}
        influence = {k: [] for k in ["pawn", "knight", "bishop", "rook", "queen", "king"]}
        annotations = {}
        
        enemy_color = "black" if color == "white" else "white"
        occupied = board_state["occupied"]
        enemy_occupied = board_state[f"{enemy_color}_pieces"]
        friendly_occupied = board_state[f"{color}_pieces"]
        is_white = color == "white"
        
        def add_move(from_sq, to_sq):
            # Register move
            if from_sq not in moves:
                moves[from_sq] = []
            moves[from_sq].append(to_sq)

            # Infer basic move type
            from_mask = 1 << from_sq
            to_mask = 1 << to_sq

            piece_key = None
            for key in board_state:
                if key.endswith("pieces") or key == "occupied":
                    continue
                if board_state[key] & from_mask:
                    piece_key = key
                    break

            # Capture check
            capture = False
            if board_state["occupied"] & to_mask:
                if (board_state[f"{'black' if color == 'white' else 'white'}_pieces"] & to_mask):
                    capture = True
            elif "pawns" in piece_key:
                # Check en passant
                if abs(from_sq - to_sq) in [7, 9] and not (board_state["occupied"] & to_mask):
                    capture = True

            annotations[(from_sq, to_sq)] = {
                "capture": capture,
                "check": False,  # will be updated later
                "mate": False
            }


        # ---- Pawns ----
        pawn_key = f"{color}_pawns"
        pawn_bb = board_state[pawn_key]
        for from_sq in Board.get_piece_squares(pawn_bb):
            targets = Board.get_pawn_targets(from_sq, board_state, last_move, is_white)
            for to_sq in targets:
                add_move(from_sq, to_sq)

            # Add diagonal squares as influence
            direction = 8 if is_white else -8
            for dx in [-1, 1]:
                col = from_sq % 8
                if 0 <= col + dx < 8:
                    diag_sq = from_sq + direction + dx
                    if 0 <= diag_sq < 64:
                        influence["pawn"].append(1 << diag_sq)

        # ---- Knights ----
        knight_bb = board_state[f"{color}_knights"]
        for from_sq in Board.get_piece_squares(knight_bb):
            move_mask, inf = Board.knight_moves(1 << from_sq, occupied, enemy_occupied)
            influence["knight"].append(inf)
            for to_sq in Board.bitboard_to_squares(move_mask & ~board_state[f"{color}_pieces"]):
                add_move(from_sq, to_sq)

        # ---- Bishops, Rooks, Queens ----
        for piece_type in ["bishop", "rook", "queen"]:
            key = f"{color}_queen" if piece_type == "queen" else f"{color}_{piece_type}s"
            bb = board_state[key]
            for from_sq in Board.get_piece_squares(bb):
                move_mask, inf = getattr(Board, f"{piece_type}_moves")(1 << from_sq, occupied, enemy_occupied)
                influence[piece_type].append(inf)
                for to_sq in Board.bitboard_to_squares(move_mask):
                    add_move(from_sq, to_sq)

        # ---- King ----
        king_bb = board_state[f"{color}_king"]
        if king_bb:
            from_sq = king_bb.bit_length() - 1
            move_mask, inf = Board.king_moves(king_bb,friendly_occupied,enemy_occupied)
            influence["king"].append(inf)
            for to_sq in Board.bitboard_to_squares(move_mask & ~board_state[f"{color}_pieces"]):
                add_move(from_sq, to_sq)

        # -- Castling Logic --
        castling_rights = board_state.get("castling_rights", {})
        occupied = board_state["occupied"]
        king_bb = board_state["white_king"] if color == "white" else board_state["black_king"]

        # Find the king's square
        if king_bb == 0:
            king_square = None
        else:
            king_square = king_bb.bit_length() - 1

        if king_square is not None:
            # Use the piece_key for move registration
            piece_key = "white_king" if color == "white" else "black_king"

            # Kingside castling
            if castling_rights.get(color, {}).get("K" if color == "white" else "k", False):
                path_clear = not (occupied & ((1 << (king_square + 1)) | (1 << (king_square + 2))))
                if path_clear:
                    add_move(king_square, king_square + 2)

            # Queenside castling
            if castling_rights.get(color, {}).get("Q" if color == "white" else "q", False):
                path_clear = not (occupied & ((1 << (king_square - 1)) | (1 << (king_square - 2)) | (1 << (king_square - 3))))
                if path_clear:
                    add_move(king_square, king_square - 2)


        return moves, influence, annotations

    @staticmethod
    def get_pawn_targets(from_sq, board_state, last_move=None, is_white=True):
        targets = []
        direction = 8 if is_white else -8
        enemy_occupied = board_state["black_pieces"] if is_white else board_state["white_pieces"]
        occupied = board_state["occupied"]

        # Forward 1
        one_step = from_sq + direction
        if 0 <= one_step < 64 and not (occupied & (1 << one_step)):
            targets.append(one_step)

            # Forward 2 from base rank
            start_rank = 1 if is_white else 6
            if from_sq // 8 == start_rank:
                two_step = from_sq + 2 * direction
                if not (occupied & (1 << two_step)):
                    targets.append(two_step)

        # Diagonal captures
        for dx in [-1, 1]:
            col = from_sq % 8
            if 0 <= col + dx < 8:
                target = from_sq + direction + dx
                if 0 <= target < 64:
                    bit = 1 << target

                    # Regular capture
                    if enemy_occupied & bit:
                        targets.append(target)

                    # En passant
                    if last_move:
                        _, from_last, to_last = last_move
                        if abs(from_last - to_last) == 16:  # 2-step move
                            en_passant_target = (from_last + to_last) // 2
                            if target == en_passant_target:
                                targets.append(target)

        return targets


    @staticmethod
    def get_piece_key_at_square(square, board_state, color):
        """Returns the piece key like 'white_knights' at the given square."""
        for suffix in ["pawns", "knights", "bishops", "rooks", "queen", "king"]:
            key = f"{color}_{suffix}"
            if board_state[key] & (1 << square):
                return key
        raise ValueError(f"No {color} piece found at square {square}")


    @staticmethod
    def is_square_threatened(square_mask, influence_list):
        return any(bb & square_mask for bb in influence_list)

    @staticmethod
    def sliding_attacks(direction, start_square, blockers, enemy_occupied):
        """Returns both moveable squares and influence in a direction from a square."""
        influence = 0
        moves = 0
        current_square = start_square
        start_file = start_square % 8

        while True:
            next_square = current_square + direction

            # Stop if off the board
            if next_square < 0 or next_square >= 64:
                break

            next_file = next_square % 8

            # Prevent wrap-around across files
            if abs(next_file - start_file) > 1:
                break

            bit = 1 << next_square
            influence |= bit  # Influence includes all seen squares

            if blockers & bit:
                if enemy_occupied & bit:
                    moves |= bit  # Can capture enemy
                break  # Blocked
            else:
                moves |= bit  # Empty square, legal move

            current_square = next_square
            start_file = next_file

        return moves, influence

    @staticmethod
    def pawn_moves(pawn_bb, empty_bb, enemy_bb, white=True, en_passant_target=None):
        moves = 0
        influence = 0

        direction = 8 if white else -8
        start_rank = 1 if white else 6
        left_attack = 7 if white else -9
        right_attack = 9 if white else -7

        bb = pawn_bb
        while bb:
            pawn = bb & -bb
            bb ^= pawn
            index = pawn.bit_length() - 1

            # One step forward
            forward = index + direction
            if 0 <= forward < 64 and (empty_bb & (1 << forward)):
                moves |= 1 << forward

                # Two steps from starting rank
                if (index // 8) == start_rank:
                    two_forward = index + 2 * direction
                    if empty_bb & (1 << two_forward):
                        moves |= 1 << two_forward
            if index == 36:
                print("on y est")
            # Captures
            for attack in [left_attack, right_attack]:
                target = index + attack
                if 0 <= target < 64 and Board.same_board_line(index, target):
                    bit = 1 << target
                    influence |= bit
                    if enemy_bb & bit:
                        moves |= bit

                    # En passant capture
                    if en_passant_target is not None and target == en_passant_target:
                        moves |= bit

        return moves, influence

    @staticmethod
    def knight_moves(knight_bb, occupied, enemy_occupied):
        # Precomputed knight jump offsets
        KNIGHT_OFFSETS = [17, 15, 10, 6, -6, -10, -15, -17]
        total_moves = 0
        total_influence = 0

        for square in range(64):
            if (knight_bb >> square) & 1:
                for offset in KNIGHT_OFFSETS:
                    target = square + offset
                    if 0 <= target < 64 and Board.same_board_line(square, target):
                        bit = 1 << target
                        total_influence |= bit
                        if not (occupied & bit) or (enemy_occupied & bit):
                            total_moves |= bit

        return total_moves, total_influence

    @staticmethod
    def same_board_line(from_sq, to_sq):
        # Prevent wrap-around on L-shaped moves
        file_diff = abs((from_sq % 8) - (to_sq % 8))
        rank_diff = abs((from_sq // 8) - (to_sq // 8))
        return file_diff <= 2 and rank_diff <= 2

    @staticmethod
    def king_moves(king_bb, friendly_occupied, enemy_occupied):
        king_moves = 0
        king_influence = 0

        king_square = king_bb.bit_length() - 1  # get square index
        for delta in [-9, -8, -7, -1, 1, 7, 8, 9]:
            target = king_square + delta
            if 0 <= target < 64 and abs((target % 8) - (king_square % 8)) <= 1:
                bit = 1 << target
                king_influence |= bit
                if not (bit & friendly_occupied):
                    king_moves |= bit

        return king_moves, king_influence

    @staticmethod
    def rook_moves(rook_bb, occupied, enemy_occupied):
        total_moves = 0
        total_influence = 0

        for square in range(64):
            if (rook_bb >> square) & 1:
                for direction in [8, -8, 1, -1]:  # Up, down, right, left
                    moves, influence = Board.sliding_attacks(direction, square, occupied, enemy_occupied)
                    total_moves |= moves
                    total_influence |= influence

        return total_moves, total_influence

    @staticmethod
    def bishop_moves(bishop_bb, occupied, enemy_occupied):
        total_moves = 0
        total_influence = 0

        for square in range(64):
            if (bishop_bb >> square) & 1:
                for direction in [9, 7, -7, -9]:
                    moves, influence = Board.sliding_attacks(direction, square, occupied, enemy_occupied)
                    moves_rep = {'white_bishops': moves}
                    influence_rep = {'white_bishops': influence}
                    occupied_rep = {'white_pawns': occupied}
                    enemy_occupied_rep = {'black_pawns': enemy_occupied}
                    total_moves |= moves
                    total_influence |= influence

        return total_moves, total_influence

    @staticmethod
    def queen_moves(queen_bb, occupied, enemy_occupied):
        total_moves = 0
        total_influence = 0

        for square in range(64):
            if (queen_bb >> square) & 1:
                for direction in [8, -8, 1, -1, 9, 7, -7, -9]:  # Rook + Bishop directions
                    moves, influence = Board.sliding_attacks(direction, square, occupied, enemy_occupied)
                    total_moves |= moves
                    total_influence |= influence

        return total_moves, total_influence

    @staticmethod
    def get_piece_squares(bb):
        """Returns list of square indices from a bitboard."""
        squares = []
        while bb:
            lsb = bb & -bb
            squares.append(lsb.bit_length() - 1)
            bb ^= lsb
        return squares

    @staticmethod
    def bitboard_to_squares(bb):
        """Converts bitboard to list of square indices."""
        return Board.get_piece_squares(bb)

    @staticmethod
    def split_bitboard(bb):
        """Splits a bitboard into a list of single-bit masks."""
        masks = []
        while bb:
            mask = bb & -bb
            masks.append(mask)
            bb ^= mask
        return masks

    # GENERAL FUNCTIONS

    def can_castle(self, color, side):
        """Returns True if castling is legal for given color and side ('k' or 'q')."""
        if color == 'w':
            king_start, rook_start = 0b00010000, (0b10000000 if side == 'q' else 0b00000001)  # e1, a1/h1
            empty_squares = 0b01100000 if side == 'k' else 0b00001110  # f1-g1 / b1-d1
            check_squares = 0b00100000 if side == 'k' else 0b00000100  # King moves f1/g1 / d1

        else:
            king_start, rook_start = 0b00010000 << 56, (0b10000000 << 56 if side == 'q' else 0b00000001 << 56)  # e8, a8/h8
            empty_squares = 0b01100000 << 56 if side == 'k' else 0b00001110 << 56  # f8-g8 / b8-d8
            check_squares = 0b00100000 << 56 if side == 'k' else 0b00000100 << 56  # f8/g8 / d8

        # Verify castling rights
        if ('K' not in self.castling_rights if color == 'w' else 'k' not in self.castling_rights):
            return False

        # Check if pieces are in the way
        if self.occupied & empty_squares:
            return False

        # Ensure no attacks on king path
        if self.is_king_in_check(color):
            return False

        opponent_attack_bb = self.get_attack_bitboard('b' if color == 'w' else 'w')
        if opponent_attack_bb & check_squares:
            return False

        return True

    def to_fen(self):
        """Converts the current board state into a FEN string."""
        board_fen = ""
        
        # Convert bitboards into piece placement (rows from 8th rank to 1st)
        for rank in range(7, -1, -1):
            empty_squares = 0
            for file in range(8):
                square = rank * 8 + file
                piece_found = False
                for piece, bb in self.bitboards.items():
                    if bb & (1 << square):
                        if empty_squares:
                            board_fen += str(empty_squares)
                            empty_squares = 0
                        board_fen += piece
                        piece_found = True
                        break
                if not piece_found:
                    empty_squares += 1
            if empty_squares:
                board_fen += str(empty_squares)
            board_fen += "/"
        
        board_fen = board_fen[:-1]  # Remove last "/"

        # Active color
        active_color = "w" if self.white_to_move else "b"

        # Castling rights
        castling_fen = "".join(k for k, v in self.castling_rights.items() if v) or "-"

        # En passant target
        en_passant_fen = "-" if self.en_passant_target == -1 else self.square_to_coord(self.en_passant_target)

        # Full FEN string
        return f"{board_fen} {active_color} {castling_fen} {en_passant_fen} 0 1"

    def from_fen(self, fen):
        """Loads a board state from a FEN string."""
        parts = fen.split()
        
        # 1. Load piece positions
        self.bitboards = {p: 0 for p in "PNBRQKpnbrqk"}
        ranks = parts[0].split("/")
        for rank in range(8):
            file = 0
            for char in ranks[7 - rank]:  # Read from rank 8 down to rank 1
                if char.isdigit():
                    file += int(char)
                else:
                    square = rank * 8 + file
                    self.bitboards[char] |= (1 << square)
                    file += 1

        # 2. Active color
        self.white_to_move = (parts[1] == "w")

        # 3. Castling rights
        self.castling_rights = {"K": "K" in parts[2], "Q": "Q" in parts[2],
                                "k": "k" in parts[2], "q": "q" in parts[2]}

        # 4. En passant target
        self.en_passant_target = -1 if parts[3] == "-" else self.coord_to_square(parts[3])

    # INDICATORS CALCULATIONS 
    

