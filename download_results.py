#!/usr/bin/env python3
"""
Helper script: prints instructions for downloading AutoDL results
and running Phase 3 assembly.
"""
print("""
═══ How to download AutoDL results ═══

Per-book output lives under work/<slug>/ (slug = PDF filename without .pdf).
Example slug for data/jiang-yun-xing.pdf: jiang-yun-xing

1. SSH into your AutoDL instance and zip the output:
   $ cd /root && zip -r wuxia_output.zip wuxia_output/

2. On your local Mac, download:
   $ scp -P <port> root@<autodl-ip>:/root/wuxia_output.zip ./
   $ unzip wuxia_output.zip -d work/<slug>/tmp_results/

3. Run Phase 3 assembly (same --book as phase 1):
   $ python main_local.py --book jiang-yun-xing.pdf --phase3

4. Verify output:
   $ ls work/<slug>/output/
   $ cat work/<slug>/output/第九回_鐵槍破犁/text.txt | head -5
""")
