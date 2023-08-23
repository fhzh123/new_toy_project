def input_to_device(batch_iter, device):

    src_sequence = batch_iter[0]
    src_att = batch_iter[1]
    trg_sequence = batch_iter[2]
    trg_att = batch_iter[3]

    src_sequence = src_sequence.to(device, non_blocking=True)
    src_att = src_att.to(device, non_blocking=True)
    trg_sequence = trg_sequence.to(device, non_blocking=True)
    trg_att = trg_att.to(device, non_blocking=True)

    return src_sequence, src_att, trg_sequence, trg_att