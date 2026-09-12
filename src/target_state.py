class TargetStateManager:
    def __init__(self, temporary_lost_frames=10, lost_frames=15, reset_frames=20):
        self.temporary_lost_frames = temporary_lost_frames
        self.lost_frames = lost_frames
        self.reset_frames = reset_frames

        self.current_target_id = None
        self.lost_counter = 0

    def choose_target(self, track_boxes):
        if not track_boxes:
            return None

        if self.current_target_id is not None:
            for box in track_boxes:
                if box["track_id"] == self.current_target_id:
                    return box

        return max(track_boxes, key=lambda box: box["confidence"])

    def update(self, track_boxes):
        target = self.choose_target(track_boxes)

        if target is not None:
            self.current_target_id = target["track_id"]
            self.lost_counter = 0
            return target, "Active", True

        if self.current_target_id is None:
            return None, "Searching", False

        self.lost_counter += 1

        if self.lost_counter <= self.temporary_lost_frames:
            return None, "Temporary Lost", False

        if self.lost_counter <= self.lost_frames:
            return None, "Lost", False

        if self.lost_counter <= self.reset_frames:
            self.current_target_id = None
            return None, "Reset", False

        self.current_target_id = None
        self.lost_counter = 0
        return None, "Searching", False
