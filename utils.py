def compute_mrr(recommendations, ground_truths):
    """
    计算 MRR（平均倒数排名）
    :param recommendations: 推荐列表（列表的列表），每个元素是一个推荐列表
    :param ground_truths: 实际相关物品列表，每个元素是对应的相关物品
    :return: MRR 值
    """
    reciprocal_ranks = []

    for rec_list, relevant_item in zip(recommendations, ground_truths):
        rank = 0
        for idx, item in enumerate(rec_list, start=1):  # 推荐列表中物品排名从 1 开始
            if item == relevant_item:
                rank = idx
                break
        if rank > 0:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)  # 如果没有相关物品，贡献值为 0

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr



