class SeverityService:
    def assign_severity(self, errors, threshold):
        labels = []

        for e in errors:
            if e <= threshold:
                labels.append("Normal")
            elif e <= 2 * threshold:
                labels.append("Low")
            elif e <= 3 * threshold:
                labels.append("Medium")
            else:
                labels.append("High")

        return labels
