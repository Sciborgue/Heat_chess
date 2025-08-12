import pygame
import pandas as pd
class Piece:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.image = None  # Chaque pièce aura son image

    
    
    def draw(self, win):
        """Dessine la pièce sur le plateau"""
        if self.image:
            x = self.col * square_size + (square_size - self.image.get_width()) // 2
            y = self.row * square_size + (square_size - self.image.get_height()) // 2
            win.blit(self.image, (x, y))

    def move(self, row, col):
        """Déplace la pièce à une nouvelle position"""
        self.row = row
        self.col = col

    def clone(self):
        """
        Creates a copy of the piece with the same attributes.
        To be overridden by subclasses to handle specific attributes.
        """
        return Piece(self.color, self.row, self.col)
    
class Pawn(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.direction = -1 if color == 'white' else 1  # Blancs avancent vers le haut (1), Noirs vers le bas (-1)
        self.has_moved = False  # Indicateur pour le premier coup
        self.just_moved_two = False  # Indicateur pour le mouvement initial de deux cases
        self.symbol = 'p'

    def __repr__(self):
        return f'Pawn("{self.color}","{self.symbol}", {self.row}, {self.col})'

    def get_valid_moves(self, board, last_move=None):
        """Retourne les mouvements valides pour un pion."""
        moves = []
        current_row, current_col = self.row, self.col
        direction = self.direction

        # Move forward
        if board.get_piece(current_row + direction, current_col) is None:
            moves.append((current_row + direction, current_col))
            # Double move on first move
            if not self.has_moved and board.get_piece(current_row + 2 * direction, current_col) is None:
                moves.append((current_row + 2 * direction, current_col))

        # Capture diagonally
        if current_col > 0:
            left_piece = board.get_piece(current_row + direction, current_col - 1)
            if left_piece and left_piece.color != self.color:
                moves.append((current_row + direction, current_col - 1))
        if current_col < 7:
            right_piece = board.get_piece(current_row + direction, current_col + 1)
            if right_piece and right_piece.color != self.color:
                moves.append((current_row + direction, current_col + 1))

        # En passant
        if last_move is not None:
            moves.extend(self.get_en_passant_moves(board, last_move))

        return moves

    def get_en_passant_moves(self, board, last_move):
        """Gère la capture en passant si possible."""
        moves = []
        if last_move is None:
            return moves

        current_row, current_col = self.row, self.col
        direction = self.direction

        # Check en passant to the left
        if current_col > 0:
            left_piece = board.get_piece(current_row, current_col - 1)
            if isinstance(left_piece, Pawn) and left_piece.color != self.color and last_move == (left_piece, current_row, current_col - 1):
                moves.append((current_row + direction, current_col - 1))

        # Check en passant to the right
        if current_col < 7:
            right_piece = board.get_piece(current_row, current_col + 1)
            if isinstance(right_piece, Pawn) and right_piece.color != self.color and last_move == (right_piece, current_row, current_col + 1):
                moves.append((current_row + direction, current_col + 1))

        return moves

    def get_influence(self, board):
        """Calcule l'influence diagonale pour le pion."""
        influence = []
        current_row, current_col = self.row, self.col
        direction = self.direction

        # Influence diagonale à gauche
        if current_col > 0:
            influence.append((current_row + direction, current_col - 1))

        # Influence diagonale à droite
        if current_col < 7:
            influence.append((current_row + direction, current_col + 1))

        return influence

    def clone(self):
        """Creates a copy of the pawn with all its attributes."""
        cloned_pawn = Pawn(self.color, self.row, self.col)
        cloned_pawn.has_moved = self.has_moved
        cloned_pawn.just_moved_two = self.just_moved_two
        return cloned_pawn

class Rook(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)  # Appelle le constructeur de base de la classe Piece
        self.symbol = 'r'  # 'r' pour la tour (rook)
        self.has_moved = False  # Indique si la tour a bougé, utile pour le roque
    def get_valid_moves(self, board):
        """Retourne les mouvements valides pour une tour (lignes et colonnes)."""
        return self.get_straight_moves(board)  
    
    def __repr__(self):
        return f'Rook("{self.color}","{self.symbol}", {self.row}, {self.col})'
    
    def get_influence(self, board):
        """Calcule l'influence verticale et horizontale pour la tour."""
        influence = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Bas, haut, droite, gauche

        for direction in directions:
            for i in range(1, 8):
                row = self.row + direction[0] * i
                col = self.col + direction[1] * i
                if 0 <= row < 8 and 0 <= col < 8:
                    piece = board.get_piece(row, col)
                    if piece is None:
                        influence.append((row, col))
                    else:
                        influence.append((row, col))
                        break
                else:
                    break
        return influence
    def clone(self):
        """
        Creates a copy of the rook with all its attributes.
        """
        cloned_rook = Rook(self.color, self.row, self.col)
        cloned_rook.has_moved = self.has_moved
        return cloned_rook
    def get_straight_moves(self, board):
        """Calcule les mouvements verticaux et horizontaux pour la tour."""
        moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Bas, haut, droite, gauche

        for direction in directions:
            for i in range(1, 8):
                row = self.row + direction[0] * i
                col = self.col + direction[1] * i
                if 0 <= row < 8 and 0 <= col < 8:
                    piece = board.get_piece(row, col)
                    if piece is None:
                        moves.append((row, col))
                    elif piece.color != self.color:
                        moves.append((row, col))
                        break
                    else:
                        break
                else:
                    break
        return moves

class Knight(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = 'n'

    def __repr__(self):
        return f'Knight("{self.color}","{self.symbol}", {self.row}, {self.col})'

    def get_valid_moves(self, board):
        """Retourne les mouvements valides pour un cavalier."""
        moves = []
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for move in knight_moves:
            row = self.row + move[0]
            col = self.col + move[1]
            if 0 <= row < 8 and 0 <= col < 8:
                piece = board.get_piece(row, col)
                if piece is None or piece.color != self.color:
                    moves.append((row, col))
        return moves

    def get_influence(self, board):
        """Calcule l'influence pour le cavalier."""
        influence = []
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for move in knight_moves:
            row = self.row + move[0]
            col = self.col + move[1]
            if 0 <= row < 8 and 0 <= col < 8:
                influence.append((row, col))
        return influence

    def clone(self):
        """Creates a copy of the knight with all its attributes."""
        return Knight(self.color, self.row, self.col)

class Bishop(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = 'b'

    def __repr__(self):
        return f'Bishop("{self.color}","{self.symbol}", {self.row}, {self.col})'

    def get_valid_moves(self, board):
        """Retourne les mouvements valides pour un fou (diagonales)."""
        return self.get_diagonal_moves(board)

    def get_influence(self, board):
        """Calcule l'influence diagonale pour le fou."""
        influence = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonales

        for direction in directions:
            for i in range(1, 8):
                row = self.row + direction[0] * i
                col = self.col + direction[1] * i
                if 0 <= row < 8 and 0 <= col < 8:
                    piece = board.get_piece(row, col)
                    if piece is None:
                        influence.append((row, col))
                    else:
                        influence.append((row, col))
                        break
                else:
                    break
        return influence

    def clone(self):
        """Creates a copy of the bishop with all its attributes."""
        return Bishop(self.color, self.row, self.col)

    def get_diagonal_moves(self, board):
        """Calcule les mouvements diagonaux pour le fou."""
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Haut-gauche, haut-droite, bas-gauche, bas-droite

        for direction in directions:
            for i in range(1, 8):  # Le fou peut se déplacer sur toute la longueur de la diagonale
                row = self.row + direction[0] * i
                col = self.col + direction[1] * i
                if 0 <= row < 8 and 0 <= col < 8:  # Assurez-vous que la case est valide
                    piece = board.get_piece(row, col)
                    if piece is None:
                        moves.append((row, col))
                    elif piece.color != self.color:  # Capture possible
                        moves.append((row, col))
                        break  # Ne pas sauter par-dessus les pièces
                    else:
                        break  # Même couleur, ne pas avancer plus loin
                else:
                    break
        return moves

class Queen(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = 'q'

    def __repr__(self):
        return f'Queen("{self.color}","{self.symbol}", {self.row}, {self.col})'

    def get_valid_moves(self, board):
        """Retourne les mouvements valides pour une reine (combinaison de diagonales et de lignes/colonnes)."""
        return self.get_diagonal_moves(board) + self.get_straight_moves(board)

    def get_influence(self, board):
        """Calcule l'influence diagonale et droite pour la reine."""
        influence = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)]  # Diagonales et lignes/colonnes

        for direction in directions:
            for i in range(1, 8):
                row = self.row + direction[0] * i
                col = self.col + direction[1] * i
                if 0 <= row < 8 and 0 <= col < 8:
                    piece = board.get_piece(row, col)
                    if piece is None:
                        influence.append((row, col))
                    else:
                        influence.append((row, col))
                        break
                else:
                    break
        return influence

    def clone(self):
        """Creates a copy of the queen with all its attributes."""
        return Queen(self.color, self.row, self.col)

    def get_diagonal_moves(self, board):
        """Appelle les mouvements diagonaux du fou."""
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for direction in directions:
            for i in range(1, 8):
                row = self.row + direction[0] * i
                col = self.col + direction[1] * i
                if 0 <= row < 8 and 0 <= col < 8:
                    piece = board.get_piece(row, col)
                    if piece is None:
                        moves.append((row, col))
                    elif piece.color != self.color:
                        moves.append((row, col))
                        break
                    else:
                        break
                else:
                    break
        return moves

    def get_straight_moves(self, board):
        """Appelle les mouvements droits de la tour."""
        moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for direction in directions:
            for i in range(1, 8):
                row = self.row + direction[0] * i
                col = self.col + direction[1] * i
                if 0 <= row < 8 and 0 <= col < 8:
                    piece = board.get_piece(row, col)
                    if piece is None:
                        moves.append((row, col))
                    elif piece.color != self.color:
                        moves.append((row, col))
                        break
                    else:
                        break
                else:
                    break
        return moves

class King(Piece):
    def __init__(self, color, row, col):
        super().__init__(color, row, col)
        self.symbol = 'k'
        self.has_moved = False

    def __repr__(self):
        return f'King("{self.color}","{self.symbol}", {self.row}, {self.col})'

    def get_valid_moves(self, board):
        moves = []
        # Ajouter les mouvements normaux du roi
        moves += self.get_king_moves(board)
        
        # Vérifie les conditions pour le roque
        if not self.has_moved:
            # Roque côté roi (petit roque)
            if self.can_castle_kingside(board):
                moves.append((self.row, self.col + 2))  # Côté droit
            
            # Roque côté dame (grand roque)
            if self.can_castle_queenside(board):
                moves.append((self.row, self.col - 2))  # Côté gauche
                
        return moves

    def get_influence(self, board):
        """Calcule l'influence pour le roi."""
        influence = []
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for move in king_moves:
            row, col = self.row + move[0], self.col + move[1]
            if 0 <= row < 8 and 0 <= col < 8:
                influence.append((row, col))
        return influence

    def clone(self):
        """Creates a copy of the king with all its attributes."""
        cloned_king = King(self.color, self.row, self.col)
        cloned_king.has_moved = self.has_moved
        return cloned_king

    def get_king_moves(self, board):
        """Retourne les mouvements du roi"""
        moves = []
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for move in king_moves:
            row, col = self.row + move[0], self.col + move[1]
            if 0 <= row < 8 and 0 <= col < 8 :
                piece = board.get_piece(row, col)
                if piece is None:
                    moves.append((row, col))
                elif piece.color != self.color:
                    moves.append((row, col))
                    break
                else:
                    break
        return moves

    def can_castle_kingside(self, board):
        # The king and rook must not have moved
        rook = board.get_piece(self.row, self.col + 3)
        if self.has_moved or not (isinstance(rook, Rook) and not rook.has_moved):
            return False

        # The squares between the king and the rook must be empty
        if any(board.get_piece(self.row, self.col + i) for i in (1, 2)):
            return False

        # The king's path must not be attacked
        threatened_squares = [(self.row, self.col), (self.row, self.col + 1), (self.row, self.col + 2)]
        if any(move[0].color != self.color and (move[1], move[2]) in threatened_squares for move in board.list_all_moves):
            return False

        return True

    def can_castle_queenside(self, board):
        # The king and rook must not have moved
        rook = board.get_piece(self.row, self.col - 4)
        if self.has_moved or not (isinstance(rook, Rook) and not rook.has_moved):
            return False

        # The squares between the king and the rook must be empty
        if any(board.get_piece(self.row, self.col - i) for i in (1, 2, 3)):
            return False

        # The king's path must not be attacked
        threatened_squares = [(self.row, self.col), (self.row, self.col -1), (self.row, self.col - 2), (self.row, self.col - 3)]
        if any(move[0].color != self.color and (move[1], move[2]) in threatened_squares for move in board.list_all_moves):
            return False

        return True
