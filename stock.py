class Toop():
    def move(self, row, col, board):
        """Déplace le pion sur une nouvelle position."""
        self.has_moved = True
        
        # Check if the move is a capture (diagonal move)
        if abs(self.col - col) == 1 and ((self.color == 'white' and row == self.row + 1) or (self.color == 'black' and row == self.row - 1)):
            target_piece = board.get_piece(row, col)
            if target_piece and target_piece.color != self.color:
                # It's a valid capture
                board.remove_piece(row, col)  # Remove the captured piece
        
        # If the pawn advances two squares, activate the en passant option
        if abs(self.row - row) == 2:
            self.just_moved_two = True
        else:
            self.just_moved_two = False

        # Update pawn's position
        self.row = row
        self.col = col

        # Promotion (if the pawn reaches the last row)
        if (self.color == 'white' and self.row == 7) or (self.color == 'black' and self.row == 0):
            self.promote()

    def promote(self):
        """Promouvoir le pion en une autre pièce (exemple : Dame)."""
        # Exemple simple : promotion automatique en Reine. Tu peux étendre pour offrir le choix.
        self.promoted_piece = Queen(self.row, self.col, self.color)
        print(f"Promotion! {self.color.capitalize()} pawn promoted to Queen.")


        def update_piece_position(self, piece):
            """Updates the position of a piece in the log, especially for kings."""
            if piece.symbol == "wk":
                self.piece_positions["white_king"] = (piece.row, piece.col)
            elif piece.symbol == "bk":
                self.piece_positions["black_king"] = (piece.row, piece.col)
            # Optionally log other pieces as well if needed for quick access

    def apply_move(self, piece, new_row, new_col):
        """Updates the board by moving a piece to a new position."""
        if (new_row, new_col) in piece.get_valid_moves(self):
            # Clear the original square
            self.grid[piece.row][piece.col] = None
            # Update the piece's position
            piece.row, piece.col = new_row, new_col
            self.grid[new_row][new_col] = piece
            piece.has_moved = True  # Track if the piece has moved (useful for castling and pawns)
            return True
        return False

    def remove_piece(self, row, col):
        """Removes a piece from the board at the given location."""
        self.board[row][col] = None

    def update_after_move(self, color):
        """Update the game state after a move is made."""
        self.valid_moves = self.generate_all_moves('white') + self.generate_all_moves('black')
        self.pawn_influence_moves = self.generate_pawn_capture_influences()
        self.influence_map = self.calculate_influence_map(self.pawn_influence_moves)

    def handle_piece_selection_and_move(self):
        """Handle user input to select pieces and moves."""
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            row, col = self.get_square_under_mouse(pos)
            if self.selected_piece:
                # Attempt to make a move if a valid destination is clicked
                self.make_move(self.selected_piece, row, col)
            else:
                # Select a new piece if it's the player's turn
                piece = self.board.get_piece(row, col)
                if piece and piece.color == self.turn:
                    self.selected_piece = piece