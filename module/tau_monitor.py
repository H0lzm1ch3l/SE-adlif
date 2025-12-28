import pytorch_lightning as pl

class TauMonitor(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        taus = ["tau_u", "tau_d", "tau_t"]
        for tau in taus:
            if hasattr(pl_module.l1, f"{tau}_trainer"):
                tau_value = getattr(pl_module.l1, f"{tau}_trainer").get_tau().detach()
                pl_module.log(f"l1_{tau}_mean", tau_value.mean().item(), prog_bar=False)
                pl_module.log(f"l1_{tau}_std", tau_value.std().item(), prog_bar=False)
            if hasattr(pl_module, "l2"):
                if hasattr(pl_module.l2, f"{tau}_trainer"):
                    tau_value = getattr(pl_module.l2, f"{tau}_trainer").get_tau().detach()
                    pl_module.log(f"l2_{tau}_mean", tau_value.mean().item(), prog_bar=False)
                    pl_module.log(f"l2_{tau}_std", tau_value.std().item(), prog_bar=False)