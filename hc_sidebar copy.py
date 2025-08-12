import pygame
import sys
import pandas as pd

class Sidebar:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)  # Rectangle de la sidebar
        self.text_lines = []  # Liste des lignes de texte (coups joués)
        self.font = pygame.font.SysFont('Arial', 16)  # Police pour afficher le texte (reduced size)
        self.input_active = False  # Indique si la barre de saisie est active
        self.input_text = ''  # Texte entré par l'utilisateur dans la barre de saisie
        self.input_box = pygame.Rect(x + 10, y + height - 40, width - 20, 30)  # Zone de saisie
        self.heatmap_enabled = True  # Heatmap switch
        self.heatmap_button = pygame.Rect(x + 10, y + height - 80, width - 20, 30)  # Heatmap switch button
        self.scroll_offset = 0  # Offset for scrolling
        self.white_control = 0
        self.black_control = 0
        self.global_influence_score = 0
        self.moves = []  # List of moves to display
        self.heatmap_alpha = 128  # Default transparency for heatmap
        self.predictions = []
        self.tree_width = 5
        self.tree_depth = 2
        self.active_input = None

    def draw(self, win, turn):
        # Draw the sidebar background
        pygame.draw.rect(win, (50, 50, 50), self.rect)

        # Draw the sections
        section_width = self.rect.width // 3
        # Section 1: Best Moves for White
        self.draw_predictions(win, self.rect.x, self.rect.y, 2*section_width, self.rect.height, turn)

        # Section 2: Parameters and Indicators
        self.draw_parameters_and_indicators(win, self.rect.x + 2 * section_width, self.rect.y, section_width, self.rect.height)

    def draw_predictions(self, win, x, y, width, height, color):
        # Draw the background
        pygame.draw.rect(win, (70, 70, 70), (x, y, width, height))
        
        # Draw the title
        title = f"Best Moves for {color.capitalize()}:"
        font = pygame.font.SysFont(None, 24)
        title_surface = font.render(title, True, (255, 255, 255))
        win.blit(title_surface, (x + 10, y + 10))
        
        # Draw the table headers
        headers = ["P", "F", "T", "Inf"]
        for depth in range(self.tree_depth):
            for idx, header in enumerate(headers):
                header_surface = font.render(f"{header} {depth + 1}", True, (255, 255, 255))
                win.blit(header_surface, (x + 10 + idx * (width // (4 * self.tree_depth)) + depth * (width // self.tree_depth), y + 40))
        
        # Draw the predictions
        y_offset = 60
        for prediction in self.predictions:
            for depth in range(self.tree_depth):
                move_text = prediction[depth * 4:(depth + 1) * 4]
                for idx, text in enumerate(move_text):
                    text_surface = font.render(text, True, (255, 255, 255))
                    win.blit(text_surface, (x + 10 + idx * (width // (4 * self.tree_depth)) + depth * (width // self.tree_depth), y + y_offset))
            y_offset += 20
            
    def draw_text(self, win, text, y_offset):
        font = pygame.font.SysFont(None, 24)
        img = font.render(text, True, (255, 255, 255))
        win.blit(img, (10, y_offset))
        
    def display_best_moves(self, win, x, y, width, height, color):
        # Draw the background
        pygame.draw.rect(win, (70, 70, 70), (x, y, width, height))
        
        # Draw the title
        title = "Best Moves for White:" if color == 'white' else "Best Moves for Black:"
        title_surface = self.font.render(title, True, (255, 255, 255))
        win.blit(title_surface, (x + 10, y + 10))
        
        # Filter and sort the moves by global influence score
        filtered_moves = [move for move in self.moves if move[0].color == color]
        if color == 'white':
            sorted_moves = sorted(filtered_moves, key=lambda move: move[5], reverse=True)
        else:
            sorted_moves = sorted(filtered_moves, key=lambda move: move[5])

        # Draw the table headers
        headers = ["Piece", "From", "To", "W Control", "B Control", "Influence"]
        for idx, header in enumerate(headers):
            header_surface = self.font.render(header, True, (255, 255, 255))
            win.blit(header_surface, (x + 10 + idx * (width // 6), y + 30))
        
        # Draw the moves list
        for idx, move in enumerate(sorted_moves[self.scroll_offset:self.scroll_offset + (height // 20)]):
            piece, row, col, wc, bc, t_inf = move
            move_text = [
                f"{piece.symbol}",
                f"{chr(piece.col + 97)}{8 - piece.row}",
                f"{chr(col + 97)}{8 - row}",
                f"{wc}",
                f"{bc}",
                f"{t_inf}"
            ]
            for jdx, text in enumerate(move_text):
                text_surface = self.font.render(text, True, (255, 255, 255))
                win.blit(text_surface, (x + 10 + jdx * (width // 6), y + 50 + 20 * idx))

    def draw_parameters_and_indicators(self, win, x, y, width, height):
        font = pygame.font.SysFont(None, 24)

        # Draw width scale
        width_label = font.render("Width:", True, (255, 255, 255))
        win.blit(width_label, (x + 10, y + 10))
        for i in range(1, 7):
            rect = pygame.Rect(x + 100 + (i - 1) * 30, y + 10, 20, 24)
            if i == self.tree_width:
                pygame.draw.rect(win, (0, 255, 0), rect)  # Highlight selected value
            else:
                pygame.draw.rect(win, (255, 255, 255), rect)
            pygame.draw.rect(win, (0, 0, 0), rect, 2)
            text = font.render(str(i), True, (0, 0, 0))
            win.blit(text, (x + 105 + (i - 1) * 30, y + 10))

        # Draw depth scale
        depth_label = font.render("Depth:", True, (255, 255, 255))
        win.blit(depth_label, (x + 10, y + 50))
        for i in range(1, 7):
            rect = pygame.Rect(x + 100 + (i - 1) * 30, y + 50, 20, 24)
            if i == self.tree_depth:
                pygame.draw.rect(win, (0, 255, 0), rect)  # Highlight selected value
            else:
                pygame.draw.rect(win, (255, 255, 255), rect)
            pygame.draw.rect(win, (0, 0, 0), rect, 2)
            text = font.render(str(i), True, (0, 0, 0))
            win.blit(text, (x + 105 + (i - 1) * 30, y + 50))

        self.width_rects = [pygame.Rect(x + 100 + (i - 1) * 30, y + 10, 20, 24) for i in range(1, 7)]
        self.depth_rects = [pygame.Rect(x + 100 + (i - 1) * 30, y + 50, 20, 24) for i in range(1, 7)]

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONUP:
            for i, rect in enumerate(self.width_rects):
                if rect.collidepoint(event.pos):
                    self.tree_width = i + 1
                    self.active_input = None
                    break
            for i, rect in enumerate(self.depth_rects):
                if rect.collidepoint(event.pos):
                    self.tree_depth = i + 1
                    self.active_input = None
                    break

        # Gestion des événements pour la barre de saisie
        if event.type == pygame.MOUSEBUTTONUP:
            # Activer la saisie si on clique sur la barre
            if self.input_box.collidepoint(event.pos):
                self.input_active = True
            else:
                self.input_active = False
            # Toggle heatmap switch
            if self.heatmap_button.collidepoint(event.pos):
                self.heatmap_enabled = not self.heatmap_enabled
            # Handle scrolling
            if event.button == 4:  # Scroll up
                self.scroll_offset = max(self.scroll_offset - 1, 0)
            elif event.button == 5:  # Scroll down
                self.scroll_offset = min(self.scroll_offset + 1, max(0, len(self.moves) - self.rect.height // 20))

        if event.type == pygame.KEYDOWN:
            if self.input_active:
                if event.key == pygame.K_RETURN:
                    self.input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                else:
                    self.input_text += event.unicode

    def update_sidebar(self, predicted_moves, win, turn):
        self.predictions = predicted_moves
        self.draw(win, turn)
        
    def draw_text(self, win, text, y_offset):
        font = pygame.font.SysFont(None, 24)
        img = font.render(text, True, (0, 0, 0))
        win.blit(img, (10, y_offset))