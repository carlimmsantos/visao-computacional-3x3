from typing import Dict

class TeamVoteManager:
    """Responsável por manter o histórico de track_ids e estabilizar as previsões usando confiança."""
    def __init__(self):
        self._team_by_track_id: Dict[int, int] = {}
        self._track_team_votes: Dict[int, Dict[int, float]] = {} 

    def get_last_known_team(self, track_id: int) -> int:
        return self._team_by_track_id.get(track_id, -1)

    def register_vote_and_get_winner(self, track_id: int, predicted_team_id: int, confidence: float) -> int:
        # Se não houver ID (ex: perda temporária do tracker), confie na predição atual
        if track_id < 0:
            return predicted_team_id

        # Acumula o voto com base na confiança
        votes = self._track_team_votes.setdefault(track_id, {})
        votes[predicted_team_id] = votes.get(predicted_team_id, 0.0) + confidence

        # Determina o vencedor histórico
        chosen_team = max(votes, key=votes.get)
        self._team_by_track_id[track_id] = chosen_team
        
        return chosen_team