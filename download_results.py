#!/usr/bin/env python3
"""
Helper script: prints instructions for downloading AutoDL results
and running Phase 3 assembly.
"""
print("""
═══ How to download AutoDL results ═══

1. SSH into your AutoDL instance and zip the output:
   $ cd /root && zip -r wuxia_output.zip wuxia_output/

2. On your local Mac, download:
   $ scp -P <port> root@<autodl-ip>:/root/wuxia_output.zip ./
   $ unzip wuxia_output.zip -d tmp_results/

3. Run Phase 3 assembly:
   $ python main_local.py --phase3

4. Verify output:
   $ ls wuxia/
   $ cat wuxia/第九回_鐵槍破犁/text.txt | head -5
""")
