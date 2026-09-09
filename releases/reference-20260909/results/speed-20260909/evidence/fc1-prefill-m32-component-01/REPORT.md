# M32 grouped FC1: rejected from serving screen

172 representative exact component comparisons passed. Compared with original baseline, paired M4096 TP-local MoE graph times rose6.01%/6.01% for K4 and7.03%/7.20% for K5. Previous M64 rowalias376 reduced this component time about16%, making M32 clearly less promising under these conditions.

Compiled K4 uses181registers/thread and27648 dynamic shared bytes; K5 uses196registers/thread and31744bytes. CUDA API reports2 blocks/SM for both, not3. Smaller tiles duplicate B reconstruction/staging per original64-row task; no gain was established. No serving test, full sweep, adoption or production restart.

Preserve original arithmetic decoder and retain M64 rowalias376 for further investigation. Next bounded decode candidate tests the existing8-warp specialization.
