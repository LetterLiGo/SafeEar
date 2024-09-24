import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
import numpy as np

def adversarial_g_loss(y_disc_gen):
    """Hinge loss"""
    loss = 0.0
    for i in range(len(y_disc_gen)):
        stft_loss = F.relu(1 - y_disc_gen[i]).mean().squeeze()
        loss += stft_loss
    return loss / len(y_disc_gen)


def feature_loss(fmap_r, fmap_gen):
    loss = 0.0
    for i in range(len(fmap_r)):
        for j in range(len(fmap_r[i])):
            stft_loss = ((fmap_r[i][j] - fmap_gen[i][j]).abs() /
                         (fmap_r[i][j].abs().mean())).mean()
            loss += stft_loss
    return loss / (len(fmap_r) * len(fmap_r[0]))


def sim_loss(y_disc_r, y_disc_gen):
    loss = 0.0
    for i in range(len(y_disc_r)):
        loss += F.mse_loss(y_disc_r[i], y_disc_gen[i])
    return loss / len(y_disc_r)

def reconstruction_loss(x, G_x, lamdba_wav=100, sr=16000, eps=1e-7):
    # NOTE (lsx): hard-coded now
    L = lamdba_wav * F.mse_loss(x, G_x)  # wav L1 loss
    # loss_sisnr = sisnr_loss(G_x, x) # 
    # L += 0.01*loss_sisnr
    # 2^6=64 -> 2^10=1024
    # NOTE (lsx): add 2^11
    for i in range(6, 12):
        # for i in range(5, 12): # Encodec setting
        s = 2**i
        melspec = MelSpectrogram(
            sample_rate=sr,
            n_fft=s,
            hop_length=s // 4,
            n_mels=64,
            wkwargs={"device": x.device}).to(x.device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        loss = ((S_x - S_G_x).abs().mean() + (
            ((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps))**2
             ).mean(dim=-2)**0.5).mean()) / i
        L += loss
    return L


def criterion_d(y_disc_r, y_disc_gen, fmap_r_det, fmap_gen_det, y_df_hat_r,
                y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g,
                fmap_s_r, fmap_s_g):
    """Hinge Loss"""
    loss = 0.0
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    for i in range(len(y_disc_r)):
        loss1 += F.relu(1 - y_disc_r[i]).mean() + F.relu(1 + y_disc_gen[
            i]).mean()
    for i in range(len(y_df_hat_r)):
        loss2 += F.relu(1 - y_df_hat_r[i]).mean() + F.relu(1 + y_df_hat_g[
            i]).mean()
    for i in range(len(y_ds_hat_r)):
        loss3 += F.relu(1 - y_ds_hat_r[i]).mean() + F.relu(1 + y_ds_hat_g[
            i]).mean()

    loss = (loss1 / len(y_disc_gen) + loss2 / len(y_df_hat_r) + loss3 /
            len(y_ds_hat_r)) / 3.0

    return loss


def criterion_g(commit_loss, x, G_x, fmap_r, fmap_gen, y_disc_r, y_disc_gen,
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r,
                y_ds_hat_g, fmap_s_r, fmap_s_g, lamdba_wav=100, lamdba_com=1000, lamdba_adv=1, lamdba_feat=1, lamdba_rec=1, sr=16000):
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    feat_loss = (feature_loss(fmap_r, fmap_gen) + sim_loss(
        y_disc_r, y_disc_gen) + feature_loss(fmap_f_r, fmap_f_g) + sim_loss(
            y_df_hat_r, y_df_hat_g) + feature_loss(fmap_s_r, fmap_s_g) +
                 sim_loss(y_ds_hat_r, y_ds_hat_g)) / 3.0
    rec_loss = reconstruction_loss(x.contiguous(), G_x.contiguous(), lamdba_wav, sr)
    total_loss = lamdba_com * commit_loss + lamdba_adv * adv_g_loss + lamdba_feat * feat_loss + lamdba_rec * rec_loss
    return total_loss, adv_g_loss, feat_loss, rec_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def adopt_dis_weight(weight, global_step, threshold=0, value=0.):
    # 0,3,6,9,13....这些时间步，不更新dis
    if global_step % 3 == 0:
        weight = value
    return weight


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, lamdba_adv=1):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(
            nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        print('last_layer cannot be none')
        assert 1 == 2
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 1.0, 1.0).detach()
    d_weight = d_weight * lamdba_adv
    return d_weight

def loss_g(codebook_loss,
           inputs,
           reconstructions,
           fmap_r,
           fmap_gen,
           y_disc_r,
           y_disc_gen,
           global_step,
           y_df_hat_r,
           y_df_hat_g,
           y_ds_hat_r,
           y_ds_hat_g,
           fmap_f_r,
           fmap_f_g,
           fmap_s_r,
           fmap_s_g,
           lamdba_wav=100, 
           lamdba_com=1000, 
           lamdba_adv=1, 
           lamdba_feat=1, 
           sr=16000,
           discriminator_iter_start=500
           ):
    """
    args:
        codebook_loss: commit loss.
        inputs: ground-truth wav.
        reconstructions: reconstructed wav.
        fmap_r: real stft-D feature map.
        fmap_gen: fake stft-D feature map.
        y_disc_r: real stft-D logits.
        y_disc_gen: fake stft-D logits.
        global_step: global training step.
        y_df_hat_r: real MPD logits.
        y_df_hat_g: fake MPD logits.
        y_ds_hat_r: real MSD logits.
        y_ds_hat_g: fake MSD logits.
        fmap_f_r: real MPD feature map.
        fmap_f_g: fake MPD feature map.
        fmap_s_r: real MSD feature map.
        fmap_s_g: fake MSD feature map.
    """
    rec_loss = reconstruction_loss(inputs.contiguous(),
                                   reconstructions.contiguous(), lamdba_wav, sr)
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    adv_mpd_loss = adversarial_g_loss(y_df_hat_g)
    adv_msd_loss = adversarial_g_loss(y_ds_hat_g)
    adv_loss = (adv_g_loss + adv_mpd_loss + adv_msd_loss
                ) / 3.0  # NOTE(lsx): need to divide by 3?
    feat_loss = feature_loss(
        fmap_r,
        fmap_gen)  #+ sim_loss(y_disc_r, y_disc_gen) # NOTE(lsx): need logits?
    feat_loss_mpd = feature_loss(fmap_f_r,
                                 fmap_f_g)  #+ sim_loss(y_df_hat_r, y_df_hat_g)
    feat_loss_msd = feature_loss(fmap_s_r,
                                 fmap_s_g)  #+ sim_loss(y_ds_hat_r, y_ds_hat_g)
    feat_loss_tot = (feat_loss + feat_loss_mpd + feat_loss_msd) / 3.0
    d_weight = torch.tensor(1.0)
    disc_factor = adopt_weight(
        lamdba_adv, global_step, threshold=discriminator_iter_start)
    if disc_factor == 0.:
        fm_loss_wt = 0
    else:
        fm_loss_wt = lamdba_feat
    loss = rec_loss + d_weight * disc_factor * adv_loss + \
           fm_loss_wt * feat_loss_tot + lamdba_com * codebook_loss
    return loss, rec_loss, adv_loss, feat_loss_tot, d_weight

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    print(thresholds[min_index])
    return eer, thresholds[min_index]