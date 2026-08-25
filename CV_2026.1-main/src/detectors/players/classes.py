"""Classes do detector de basquete (dataset basquete_combined, 11 classes).

Fonte única para nomes e ids de classe usados por pipeline, scripts e
módulos de ação — evita listas/dicionários duplicados pelo código.
"""

from typing import Dict

BALL = 0
BALL_IN_BASKET = 1
NUMBER = 2
PLAYER = 3
PLAYER_IN_POSSESSION = 4
PLAYER_JUMP_SHOT = 5
PLAYER_LAYUP_DUNK = 6
PLAYER_SHOT_BLOCK = 7
PLAYER_BOX_OUT = 8
REFEREE = 9
RIM = 10

CLASS_NAMES: Dict[int, str] = {
    BALL: "ball",
    BALL_IN_BASKET: "ball-in-basket",
    NUMBER: "number",
    PLAYER: "player",
    PLAYER_IN_POSSESSION: "player-in-possession",
    PLAYER_JUMP_SHOT: "player-jump-shot",
    PLAYER_LAYUP_DUNK: "player-layup-dunk",
    PLAYER_SHOT_BLOCK: "player-shot-block",
    PLAYER_BOX_OUT: "player-box-out",
    REFEREE: "referee",
    RIM: "rim",
}

# Classes de jogador relevantes para pose/arremesso
PLAYER_CLASS_IDS = {
    PLAYER,
    PLAYER_IN_POSSESSION,
    PLAYER_JUMP_SHOT,
    PLAYER_LAYUP_DUNK,
    PLAYER_SHOT_BLOCK,
}

# Team assignment também classifica jogadores em box-out
TEAM_PLAYER_CLASS_IDS = PLAYER_CLASS_IDS | {PLAYER_BOX_OUT}

# Classes que o detector já rotula como ação de arremesso (prior fraco)
SHOOTING_CLASS_IDS = {PLAYER_JUMP_SHOT, PLAYER_LAYUP_DUNK}
