import pygame
def square_to_algebraic(square):
    col = square % 8
    row = square // 8
    file = chr(ord('a') + col)
    rank = str(row + 1)
    return file + rank


class Sidebar:
    def __init__(self, game, width, height):
        self.game = game
        self.scroll_offset = 0
        self.width = width
        self.height = height
        

        self.font = pygame.font.SysFont('Arial', 16)
        self.x = 800  # Assuming 8x8 board at 100px per square
        self.y = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.input_rect = pygame.Rect(self.x + 10, self.height - 40, 200, 30)
        self.symbolic_moves = []

        # Tree parameters
        self.tree_width = 5
        self.tree_depth = 2

        self.predictions = []
        self.turn = "white"
        self.symbolic_moves = []

        self.indicators = {} 

        # Control for method weights
        self.method_weights = {
            "heatmap": 1,
            "belt": 0,
            "integrity": 0,
            "squares": 0,
            "king_belt": 0,
            "center": 0
        }
        self.method_enabled = {
            "heatmap": True,
            "belt": True,
            "integrity": True,
            "squares": True,
            "king_belt": True,
            "center": True
        }

        self.method_weight_buttons = {}
        start_y = self.y + 30
        for i, method in enumerate(self.method_weights):
            base_y = start_y + i * 30
            self.method_weight_buttons[method] = {
                "decrease": pygame.Rect(self.x + 300, base_y, 20, 20),
                "increase": pygame.Rect(self.x + 360, base_y, 20, 20),
                "reset": pygame.Rect(self.x + 420, base_y, 20, 20)
            }

        # Tree width/depth buttons
        settings_x = self.x + 10
        self.width_decrease_rect = pygame.Rect(settings_x + 60, self.y + 30, 20, 20)
        self.width_increase_rect = pygame.Rect(settings_x + 120, self.y + 30, 20, 20)
        self.depth_decrease_rect = pygame.Rect(settings_x + 60, self.y + 60, 20, 20)
        self.depth_increase_rect = pygame.Rect(settings_x + 120, self.y + 60, 20, 20)

        # Score layer buttons ("S" buttons)
        self.score_buttons = [
            pygame.Rect(self.x + 50 + i * 50, self.y + 5, 30, 25)
            for i in range(4)
        ]

        self.active_score_layer = 0  # 0: None, 1: Heatmap, 2: Square Possession, etc.

        # Display toggles
        self.show_heatmap = True  # default to visible

        # Input and move log field
        self.input_active = False
        self.input_text = ""
        self.input_box = pygame.Rect(self.x + 10, self.y + self.height - 40, 200, 30)

        # Placeholder for future clickable buttons
        self.buttons = []


    
    def format_move_log(move_log):
        lines = []
        for i in range(0, len(move_log), 2):
            move_num = i // 2 + 1
            white_move = move_log[i][0] if i < len(move_log) else ""
            black_move = move_log[i + 1][0] if i + 1 < len(move_log) else ""
            lines.append(f"{move_num}. {white_move:<6} {black_move}")
        return "\n".join(lines)

    @staticmethod
    def square_to_algebraic(square):
        file = square % 8
        rank = square // 8
        return f"{chr(ord('a') + file)}{8 - rank}"

    @staticmethod
    def algebraic_to_square(algebraic):
        """Convert algebraic notation like 'e4' into a 0–63 square index."""
        if len(algebraic) != 2:
            return None
        file_char = algebraic[0].lower()
        rank_char = algebraic[1]
        if file_char < 'a' or file_char > 'h' or rank_char < '1' or rank_char > '8':
            return None
        file = ord(file_char) - ord('a')
        rank = 8 - int(rank_char)
        return rank * 8 + file


    
    def display_best_moves(self, win, x, y, width, height, color, predictions_list):
        "Tree setting"
        # Tree Settings 
        font = pygame.font.SysFont(None, 22)
        win.blit(font.render("TREE", True, (255, 255, 255)), (x + 10, y + 5))

        win.blit(font.render("Width:", True, (220, 220, 220)), (x + 10, y + 30))
        pygame.draw.rect(win, (100, 100, 100), self.width_decrease_rect)
        win.blit(font.render("-", True, (255, 255, 255)), (self.width_decrease_rect.x + 5, self.width_decrease_rect.y))
        win.blit(font.render(str(self.tree_width), True, (255, 255, 255)), (self.width_decrease_rect.right + 10, self.width_decrease_rect.y))
        pygame.draw.rect(win, (100, 100, 100), self.width_increase_rect)
        win.blit(font.render("+", True, (255, 255, 255)), (self.width_increase_rect.x + 5, self.width_increase_rect.y))

        win.blit(font.render("Depth:", True, (220, 220, 220)), (x + 10, y + 60))
        pygame.draw.rect(win, (100, 100, 100), self.depth_decrease_rect)
        win.blit(font.render("-", True, (255, 255, 255)), (self.depth_decrease_rect.x + 5, self.depth_decrease_rect.y))
        win.blit(font.render(str(self.tree_depth), True, (255, 255, 255)), (self.depth_decrease_rect.right + 10, self.depth_decrease_rect.y))
        pygame.draw.rect(win, (100, 100, 100), self.depth_increase_rect)
        win.blit(font.render("+", True, (255, 255, 255)), (self.depth_increase_rect.x + 5, self.depth_increase_rect.y))
        
        for i, rect in enumerate(self.width_rects):
            pygame.draw.rect(win, (0, 255, 0) if self.tree_width == i + 1 else (255, 255, 255), rect)
            win.blit(self.font.render(str(i + 1), True, (0, 0, 0)), (rect.x + 4, rect.y + 2))

        for i, rect in enumerate(self.depth_rects):
            pygame.draw.rect(win, (0, 255, 0) if self.tree_depth == i + 1 else (255, 255, 255), rect)
            win.blit(self.font.render(str(i + 1), True, (0, 0, 0)), (rect.x + 4, rect.y + 2))
            
        pygame.draw.rect(win, (70, 70, 70), (x, y, width, height))
        title = "Best Moves for White:" if color == 'white' else "Best Moves for Black:"
        win.blit(self.font.render(title, True, (255, 255, 255)), (x + 10, y + 10))

        headers = ["P", "Fr", "To", "Sc"]
        for i, h in enumerate(headers):
            win.blit(self.font.render(h, True, (255, 255, 255)), (x + 10 + i * (width // 6), y + 30))

        for idx, move in enumerate(predictions_list[self.scroll_offset:self.scroll_offset + 10]):
            print(idx,move)
            piece, from_square, to_square, score = move[0], move[1], move[2], move[3]
            move_data = [
                piece["type"][6].upper(),  # "white_pawn" → "P"
                square_to_algebraic(from_square),
                square_to_algebraic(to_square),
                f"{score:.1f}"
            ]
            for j, text in enumerate(move_data):
                win.blit(self.font.render(text, True, (255, 255, 255)), (x + 10 + j * (width // 6), y + 50 + idx * 20))

    def display_available_moves(self, win, x, y, width, height):
        pygame.draw.rect(win, (50, 50, 50), (x, y, width, height))
        title = self.font.render("Available Moves", True, (255, 255, 255))
        win.blit(title, (x + 10, y + 5))

        row_height = 20
        move_width = 80
        cols = max(1, width // move_width)
        rows = max(1, (height - 30) // row_height)
        max_moves = cols * rows

        displayed = self.symbolic_moves[self.scroll_offset:self.scroll_offset + max_moves]

        for i, move_text in enumerate(displayed):
            col_i = i % cols
            row_i = i // cols
            x_pos = x + col_i * move_width
            y_pos = y + 30 + row_i * row_height
            win.blit(self.font.render(move_text, True, (255, 255, 255)), (x_pos, y_pos))


    def adjust_method_weight(self, method, delta):
        """Adjusts integer weights. Resets if delta is 'reset'."""
        if delta == "reset":
            self.method_weights[method] = 1
        else:
            self.method_weights[method] = max(0, self.method_weights[method] + delta)

    def draw(self, win, turn, predictions):
        """Draw the full sidebar interface, passing dynamic game context."""
        pygame.draw.rect(win, (30, 30, 30), self.rect)  # Sidebar background

        # Dimensions and layout tracking
        top = self.y
        section_margin = 10

        # HEADER: Tree controls, method weights, score buttons
        header_height = 180
        self.draw_header(win, self.x, top, self.width, header_height)
        top += header_height + section_margin

        # STATUS: Heatmap control, belt, square possession, etc.
        status_height = 160
        self.draw_status_panel(win, self.x, top, self.width, status_height)
        top += status_height + section_margin

        # PREDICTIONS: Move suggestions from the engine
        predictions_height = int(self.height * 0.3)
        self.draw_predictions(win, self.x, top, self.width, predictions_height, predictions, turn)
        top += predictions_height + section_margin

        # AVAILABLE MOVES: All possible moves or selected piece
        available_moves_height = self.height - top - 60
        self.draw_available_moves(win, self.x, top, self.width, available_moves_height)
        top += available_moves_height + section_margin

        # FOOTER: Input box for manual move input + move log
        footer_height = 50
        self.draw_footer(win, self.x, self.y + self.height - footer_height, self.width, footer_height)

    def draw_predictions(self, win, x, y, width, height, predictions, turn):
        """Draws the prediction list passed from Game."""
        pygame.draw.rect(win, (35, 35, 35), (x, y, width, height))
        title = self.font.render(f"Best Moves for {turn.title()}", True, (255, 255, 255))
        win.blit(title, (x + 10, y + 5))

        row_height = 20
        padding_top = 30
        max_rows = (height - padding_top) // row_height

        for i, (piece, from_sq, to_sq, score) in enumerate(predictions[:max_rows]):
            from_alg = self.square_to_algebraic(from_sq)
            to_alg = self.square_to_algebraic(to_sq)
            symbol = self.format_piece_letter(piece["type"])
            move_str = f"{symbol}{from_alg} → {to_alg} ({score:.2f})"

            text = self.font.render(move_str, True, (230, 230, 230))
            win.blit(text, (x + 10, y + padding_top + i * row_height))

    @staticmethod
    def square_to_algebraic(square):
        file = square % 8
        rank = square // 8
        return f"{chr(ord('a') + file)}{8 - rank}"

    def draw_available_moves(self, win, x, y, width, height):
        """Draws the available symbolic moves provided by the game."""
        pygame.draw.rect(win, (45, 45, 45), (x, y, width, height))
        title = self.font.render("Available Moves", True, (255, 255, 255))
        win.blit(title, (x + 10, y + 5))

        row_height = 20
        col_width = 100
        padding_top = 30

        cols = max(1, width // col_width)
        rows = max(1, (height - padding_top) // row_height)
        max_moves = cols * rows

        display_moves = self.symbolic_moves[self.scroll_offset:self.scroll_offset + max_moves]

        for idx, move in enumerate(display_moves):
            col = idx % cols
            row = idx // cols
            text = self.font.render(move, True, (220, 220, 220))
            win.blit(text, (x + col * col_width + 10, y + padding_top + row * row_height))

    def format_piece_letter(self, piece_type):
        return {
            "pawns": "",
            "knights": "N",
            "bishops": "B",
            "rooks": "R",
            "queen": "Q",
            "king": "K"
        }.get(piece_type.replace("white_", "").replace("black_", ""), "?")

    def draw_header(self, win, x, y, width, height):
        font = pygame.font.SysFont(None, 22)
        pygame.draw.rect(win, (30, 30, 30), (x, y, width, height))

        # Layout width zones
        tree_w = width // 5
        weights_w = (width * 3) // 5
        layers_w = width - tree_w - weights_w

        # ==== TREE SETTINGS (Left side) ====
        win.blit(font.render("TREE", True, (255, 255, 255)), (x + 10, y + 5))

        win.blit(font.render("Width:", True, (220, 220, 220)), (x + 10, y + 30))
        pygame.draw.rect(win, (100, 100, 100), self.width_decrease_rect)
        win.blit(font.render("-", True, (255, 255, 255)), (self.width_decrease_rect.x + 5, self.width_decrease_rect.y))
        win.blit(font.render(str(self.tree_width), True, (255, 255, 255)), (self.width_decrease_rect.right + 10, self.width_decrease_rect.y))
        pygame.draw.rect(win, (100, 100, 100), self.width_increase_rect)
        win.blit(font.render("+", True, (255, 255, 255)), (self.width_increase_rect.x + 5, self.width_increase_rect.y))

        win.blit(font.render("Depth:", True, (220, 220, 220)), (x + 10, y + 60))
        pygame.draw.rect(win, (100, 100, 100), self.depth_decrease_rect)
        win.blit(font.render("-", True, (255, 255, 255)), (self.depth_decrease_rect.x + 5, self.depth_decrease_rect.y))
        win.blit(font.render(str(self.tree_depth), True, (255, 255, 255)), (self.depth_decrease_rect.right + 10, self.depth_decrease_rect.y))
        pygame.draw.rect(win, (100, 100, 100), self.depth_increase_rect)
        win.blit(font.render("+", True, (255, 255, 255)), (self.depth_increase_rect.x + 5, self.depth_increase_rect.y))

        # ==== METHOD WEIGHTS (Middle, 2 columns) ====
        method_x = x + tree_w
        win.blit(font.render("METHOD WEIGHTS", True, (255, 255, 255)), (method_x + 10, y + 5))

        col_spacing = 200
        col1_x = method_x + 10
        col2_x = method_x + 10 + col_spacing
        method_spacing = 28

        methods = list(self.method_weights.keys())
        for i, method in enumerate(methods):
            col_x = col1_x if i % 2 == 0 else col2_x
            row_y = y + 30 + (i // 2) * method_spacing

            win.blit(font.render(method.replace("_", " ").title(), True, (220, 220, 220)), (col_x, row_y))

            dec = self.method_weight_buttons[method]["decrease"]
            inc = self.method_weight_buttons[method]["increase"]
            reset = self.method_weight_buttons[method]["reset"]

            dec.x, dec.y = col_x + 80, row_y
            inc.x, inc.y = col_x + 130, row_y
            reset.x, reset.y = col_x + 155, row_y

            pygame.draw.rect(win, (100, 100, 100), dec)
            win.blit(font.render("-", True, (255, 255, 255)), (dec.x + 5, dec.y))

            pygame.draw.rect(win, (100, 100, 100), inc)
            win.blit(font.render("+", True, (255, 255, 255)), (inc.x + 5, inc.y))

            pygame.draw.rect(win, (100, 100, 100), reset)
            win.blit(font.render("0", True, (255, 255, 255)), (reset.x + 5, reset.y))

            # Draw current weight
            value_text = font.render(str(self.method_weights[method]), True, (255, 255, 255))
            win.blit(value_text, (dec.right + 10, dec.y))

        
        # ==== SCORE LAYERS (Right side) ====
        score_x = x + tree_w + weights_w
        win.blit(font.render("SCORE LAYERS", True, (255, 255, 255)), (score_x + 10, y + 5))

        score_labels = ["S1: Heatmap", "S2: Squares", "S3: King Belt", "S4: Center"]
        for i, rect in enumerate(self.score_buttons):
            rect.x = score_x + 20
            rect.y = y + 35 + i * 30
            pygame.draw.rect(win, (100, 100, 100), rect)
            label = font.render(score_labels[i], True, (255, 255, 255))
            win.blit(label, (rect.right + 10, rect.y + 3))

    def draw_status_panel(self, win, x, y, width, height):
        font = pygame.font.SysFont(None, 22)
        pygame.draw.rect(win, (40, 40, 40), (x, y, width, height))

        title = font.render("CURRENT SITUATION", True, (255, 255, 255))
        win.blit(title, (x + 10, y + 5))

        col_width = width // 3
        col_xs = [x + col_width * i for i in range(3)]
        offset_y = y + 30
        spacing = 20

        data = self.indicators  # Ensure indicators are updated from Game

        # Column 1: Control
        win.blit(font.render("Control", True, (180, 180, 180)), (col_xs[0], offset_y))
        win.blit(font.render(f"W: {data['white_control']}", True, (220, 220, 220)), (col_xs[0], offset_y + spacing))
        win.blit(font.render(f"B: {data['black_control']}", True, (220, 220, 220)), (col_xs[0], offset_y + spacing * 2))
        win.blit(font.render(f"Net: {data['global_influence_score']}", True, (220, 220, 220)), (col_xs[0], offset_y + spacing * 3))

        # Column 2: Square Possession
        win.blit(font.render("Squares", True, (180, 180, 180)), (col_xs[1], offset_y))
        win.blit(font.render(f"W: {data['white_squares']}", True, (220, 220, 220)), (col_xs[1], offset_y + spacing))
        win.blit(font.render(f"B: {data['black_squares']}", True, (220, 220, 220)), (col_xs[1], offset_y + spacing * 2))
        win.blit(font.render(f"N: {data['total_squares']}", True, (220, 220, 220)), (col_xs[1], offset_y + spacing * 3))

        # Column 3: King Belt + Center
        win.blit(font.render("King Belt", True, (180, 180, 180)), (col_xs[2], offset_y))
        win.blit(font.render(f"W: {data['white_belt_score']} ({data['white_threat_count']})", True, (220, 220, 220)), (col_xs[2], offset_y + spacing))
        win.blit(font.render(f"B: {data['black_belt_score']} ({data['black_threat_count']})", True, (220, 220, 220)), (col_xs[2], offset_y + spacing * 2))
        win.blit(font.render(f"Center T:{data['tight_center_score']} L:{data['large_center_score']}", True, (200, 200, 200)), (col_xs[2], offset_y + spacing * 3))

    def draw_footer(self, win, x, y, width, height):
        """Draws the input box and move log at the bottom of the sidebar."""
        pygame.draw.rect(win, (30, 30, 30), (x, y, width, height))
        font = pygame.font.SysFont(None, 22)

        # Move Input Box on the left
        input_x = x + 10
        self.input_rect.topleft = (input_x, y + 30)
        pygame.draw.rect(win, (60, 60, 60), self.input_rect, 2)
        txt_surface = font.render(self.input_text, True, (255, 255, 255))
        win.blit(txt_surface, (self.input_rect.x + 5, self.input_rect.y + 5))

        win.blit(font.render("Enter Move(s):", True, (255, 255, 255)), (input_x, y + 5))

        # Move Log on the right
        log_x = self.input_rect.right + 30
        log_y = y + 5
        line_height = 22
        max_lines = (height - 10) // line_height
        move_log = self.game.move_log[-max_lines:]

        for i, (color, piece, from_sq, to_sq, score) in enumerate(move_log):
            move_text = self.game.format_algebraic_move(from_sq, to_sq, piece["type"])
            win.blit(font.render(f"{color[0].upper()}: {move_text}", True, (220, 220, 220)), (log_x, log_y + i * line_height))



    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONUP:
            # Method weight buttons
            for method, buttons in self.method_weight_buttons.items():
                if buttons["decrease"].collidepoint(event.pos):
                    if self.method_weights[method] > 0:
                        self.adjust_method_weight(method, -1)
                    return
                elif buttons["increase"].collidepoint(event.pos):
                    self.adjust_method_weight(method, 1)
                    return
                elif buttons["reset"].collidepoint(event.pos):
                    self.method_weights[method] = 1
                    return

            # Score toggle buttons (S1, S2, S3, S4)
            for i, button in enumerate(self.score_buttons):
                if button.collidepoint(event.pos):
                    self.active_score_layer = i + 1  # Show layer 1–4
                    return

            # Input box click
            if self.input_rect.collidepoint(event.pos):
                self.input_active = True
            else:
                self.input_active = False

        elif event.type == pygame.KEYDOWN and self.input_active:
            if event.key == pygame.K_RETURN:
                self.submit_move_input()  # Submit entered move(s)
                self.input_text = ""
                self.input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode

    def update_sidebar(self, predictions, win, turn, symbolic_moves, **indicators):
        self.predictions = predictions
        self.turn = turn
        self.symbolic_moves = symbolic_moves
        self.indicators = indicators  # Stores all keys like 'white_control', 'tight_center_score', etc.


