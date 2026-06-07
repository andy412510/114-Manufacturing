class ACE_Playbook:
    def __init__(self):
        self.playbook = []

    def generator(self, trend, top5_lags, expert_knowledge):
        return (
            f"[ACE 動態上下文]\n"
            f"趨勢：{trend}\n"
            f"前五大滯後：{top5_lags}\n"
            f"專家知識：{expert_knowledge}\n"
            f"建議：溫度與壓力對位移影響大。"
        )

    def curator(self, new_strategy):
        self.playbook.append(new_strategy)
        return self.playbook[-3:]
