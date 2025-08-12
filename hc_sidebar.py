import pygame
import sys
import pandas as pd

class Sidebar:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height        
        self.rect = pygame.Rect(x, y, width, height)  # Rectangle de la sidebar
        self.text_lines = []  # Liste des lignes de texte (coups joués)
        self.font = pygame.font.SysFont('Arial', 16)  # Police pour afficher le texte (reduced size)
        self.input_active = False  # Indique si la barre de saisie est active
        self.input_text = ''  # Texte entré par l'utilisateur dans la barre de saisie
        self.input_box = pygame.Rect(x + 10, y + height - 40, width - 20, 30)  # Zone de saisie
        self.methodology = 'heatmap'  # Default methodology
        self.heatmap_button = pygame.Rect(x + 10, y + height - 80, width - 20, 30)  # Heatmap switch button
        self.belt_button = pygame.Rect(x + 10, y + height - 120, width - 20, 30)  # King belt switch button
        self.integrity_button = pygame.Rect(x + 10, y + height - 160, width - 20, 30)  # Pieces integrity switch button
        self.scroll_offset = 0  # Offset for scrolling
        self.white_control = 0
        self.black_control = 0
        self.global_influence_score = 0
        self.tree_width = 5
        self.tree_depth = 2
        self.width_rects = [pygame.Rect(x + 100 + (i - 1) * 30, y + 10, 20, 24) for i in range(1, 7)]
        self.depth_rects = [pygame.Rect(x + 100 + (i - 1) * 30, y + 50, 20, 24) for i in range(1, 7)]

    def draw(self, win, turn, predictions_list):
        # Draw the sidebar background
        pygame.draw.rect(win, (0, 0, 0), self.rect)
        
        # Calculate panel dimensions
        panel_width = self.rect.width * 2 // 3
        panel_height = self.rect.height
        info_panel_width = self.rect.width // 3
        info_panel_height = self.rect.height // 3
        
        self.integrity_button = pygame.Rect(self.x + 5 + panel_width, self.y + self.height - 80, info_panel_width - 10, 30) 
        self.belt_button = pygame.Rect(self.x + 5 + panel_width, self.y + self.height - 120, info_panel_width - 10, 30) 
        self.heatmap_button = pygame.Rect(self.x + 5 + panel_width, self.y + self.height - 160, info_panel_width - 10, 30) 
        self.width_rects = [pygame.Rect(self.x + panel_width + 100 + (i - 1) * 30, self.y + info_panel_height + 10, 20, 24) for i in range(1, 7)]
        self.depth_rects = [pygame.Rect(self.x + panel_width + 100 + (i - 1) * 30, self.y + info_panel_height + 50, 20, 24) for i in range(1, 7)]
        # Draw the best moves panel
        self.display_best_moves(win, self.rect.x, self.rect.y, panel_width, panel_height, turn, predictions_list)
        
        # Draw the info panel
        self.draw_info_panel(win, self.rect.x + panel_width, self.rect.y, info_panel_width, info_panel_height)
        
        # Draw the manual move input box
        self.draw_input_box(win, self.rect.x + panel_width, self.rect.y + info_panel_height, info_panel_width, info_panel_height)
        
        # Draw the tree settings panel
        self.draw_tree_settings_panel(win, self.rect.x + panel_width, self.rect.y + info_panel_height, info_panel_width, info_panel_height)
        
        # Draw the methodology panel
        self.draw_methodology_panel(win, self.rect.x + panel_width, self.rect.y + 2 * info_panel_height, info_panel_width, info_panel_height)
    
    def display_best_moves(self, win, x, y, width, height, color, predictions_list):
        # Draw the background
        pygame.draw.rect(win, (70, 70, 70), (x, y, width, height))
        
        # Draw the title
        title = "Best Moves for White:" if color == 'white' else "Best Moves for Black:"
        title_surface = self.font.render(title, True, (255, 255, 255))
        win.blit(title_surface, (x + 10, y + 10))
        
        sorted_moves = predictions_list
        
        # Draw the table headers
        headers = ["P", "Fr", "To", "Sc"]
        for idx, header in enumerate(headers):
            header_surface = self.font.render(header, True, (255, 255, 255))
            win.blit(header_surface, (x + 10 + idx * (width // 6), y + 30))
        
        # Draw the moves list
        for idx, move in enumerate(sorted_moves[self.scroll_offset:self.scroll_offset + (height // 20)]):
            piece, row, col, t_inf = move
            move_text = [
                f"{piece.symbol}",
                f"{chr(piece.col + 97)}{8 - piece.row}",
                f"{chr(col + 97)}{8 - row}",
                f"{t_inf}"
            ]
            for jdx, text in enumerate(move_text):
                text_surface = self.font.render(text, True, (255, 255, 255))
                win.blit(text_surface, (x + 10 + jdx * (width // 6), y + 50 + idx * 20))
    
    def draw_input_box(self, win, x, y, width, height):
        # Draw the background
        pygame.draw.rect(win, (0, 0, 0), (x, y, width, height))  # Light grey background
        pygame.draw.rect(win, (40, 40, 40), (x + 5, y + 5, width - 10, height - 10))
        
        # Draw the input box
        pygame.draw.rect(win, (255, 255, 255) if self.input_active else (200, 200, 200), self.input_box)
        text_surface = self.font.render(self.input_text, True, (0, 0, 0))
        win.blit(text_surface, (x + 10, y + 10))
        
                
    def draw_info_panel(self, win, x, y, width, height):
        # Draw the control and influence scores
        white_control_text = self.font.render(f"White Control: {self.white_control}", True, (255, 255, 255))
        win.blit(white_control_text, (x + 10, y + 10))
        
        black_control_text = self.font.render(f"Black Control: {self.black_control}", True, (255, 255, 255))
        win.blit(black_control_text, (x + 10, y + 30))
        
        influence_score_text = self.font.render(f"Global Influence: {self.global_influence_score}", True, (255, 255, 255))
        win.blit(influence_score_text, (x + 10, y + 50))

    def draw_tree_settings_panel(self, win, x, y, width, height):
        # Draw the background
        pygame.draw.rect(win, (0, 0, 0), (x, y, width, height))  # Light grey background
        pygame.draw.rect(win, (40, 40, 40), (x + 5, y + 5, width - 10, height - 10))  # Dark grey inner rectangle
        
        # Draw the tree width and depth
        for i, rect in enumerate(self.width_rects):
            pygame.draw.rect(win, (0, 255, 0) if self.tree_width == i + 1 else (255, 255, 255), rect)
            pygame.draw.rect(win, (0, 0, 0), rect, 2)
            text = self.font.render(str(i + 1), True, (0, 0, 0))
            win.blit(text, (x + 5, y + 5))
        
        for i, rect in enumerate(self.depth_rects):
            pygame.draw.rect(win, (0, 255, 0) if self.tree_depth == i + 1 else (255, 255, 255), rect)
            pygame.draw.rect(win, (0, 0, 0), rect, 2)
            text = self.font.render(str(i + 1), True, (0, 0, 0))
            win.blit(text, (x + 5, y + 5))

    def draw_methodology_panel(self, win, x, y, width, height):
        # Draw the background
        pygame.draw.rect(win, (0, 0, 0), (x, y, width, height))  # Light grey background
        pygame.draw.rect(win, (60, 60, 60), (x + 5, y + 5, width - 10, height - 10))  # Dark grey inner rectangle
        
        # Draw the buttons
        pygame.draw.rect(win, (255, 0, 0) if self.methodology == 'heatmap' else (200, 200, 200), self.heatmap_button)
        heatmap_text = self.font.render("Heat", True, (0, 0, 0))
        win.blit(heatmap_text, (self.heatmap_button.x + 5, self.heatmap_button.y + 5))
        
        pygame.draw.rect(win, (255, 0, 0) if self.methodology == 'belt' else (200, 200, 200), self.belt_button)
        belt_text = self.font.render("Belt", True, (0, 0, 0))
        win.blit(belt_text, (self.belt_button.x + 5, self.belt_button.y + 5))
        
        pygame.draw.rect(win, (255, 0, 0) if self.methodology == 'integrity' else (200, 200, 200), self.integrity_button)
        integrity_text = self.font.render("Integrity", True, (0, 0, 0))
        win.blit(integrity_text, (self.integrity_button.x + 5, self.integrity_button.y + 5))

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
            # Activer la saisie si on clique sur la barre
            if self.input_box.collidepoint(event.pos):
                self.input_active = True
            else:
                self.input_active = False
            # Toggle heatmap switch
            if self.heatmap_button.collidepoint(event.pos):
                self.methodology = 'heatmap'
            # Toggle king belt switch
            if self.belt_button.collidepoint(event.pos):
                self.methodology = 'belt'
            # Toggle pieces integrity switch
            if self.integrity_button.collidepoint(event.pos):
                self.methodology = 'integrity'
            # Handle scrolling
            if event.button == 4:  # Scroll up
                self.scroll_offset = max(self.scroll_offset - 1, 0)
            elif event.button == 5:  # Scroll down
                self.scroll_offset = min(self.scroll_offset + 1, max(0, len(self.text_lines) - self.rect.height // 20))
        if event.type == pygame.KEYDOWN:
            if self.input_active:
                if event.key == pygame.K_RETURN:
                    self.input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                else:
                    self.input_text += event.unicode

    def update_sidebar(self, predicted_moves, win, turn, white_control, black_control, global_influence_score):
        self.predictions = predicted_moves
        self.white_control = white_control
        self.black_control = black_control
        self.global_influence_score = global_influence_score
        self.draw(win, turn, predicted_moves)

    