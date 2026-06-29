Issue 3b850d6 audits the closed-loop finite-policy soft-adversary path on the
existing c92 frozen no-PGD substrates. It separates raw policy outputs from
diagnostic cap clipping and tests known affine and linear policy directions
derived from the direct-epsilon sweep, without updating controller weights or
launching training.
