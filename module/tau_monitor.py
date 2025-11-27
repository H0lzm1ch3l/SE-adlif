import pytorch_lightning as pl

class TauMonitor(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        tau_u = pl_module.l2.tau_u_trainer.get_tau()
        tau_d = pl_module.l2.tau_d_trainer.get_tau()
        tau_t = pl_module.l2.tau_t_trainer.get_tau()
        pl_module.log("tau_u", tau_u.mean().item(), prog_bar=True)
        pl_module.log("tau_d", tau_d.mean().item(), prog_bar=True)
        pl_module.log("tau_t", tau_t.mean().item(), prog_bar=True)
        # variance
        pl_module.log("tau_u_var", tau_u.var().item(), prog_bar=False)
        pl_module.log("tau_d_var", tau_d.var().item(), prog_bar=False)
        pl_module.log("tau_t_var", tau_t.var().item(), prog_bar=False)

    