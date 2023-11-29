'''
Minimal test, starting stella_vslam.
It tests import and some basic bindings.
You can see stella_vslam console messages while warming up and shutting down.
You need equirectangular.yaml and orb_vocab.fbow.  Feel free to change their paths.
'''

import stellavslam as vslam

print("This is an operational test.  It should open stellavslam binding module, init and shutdown stellavslam without errors,")
print("and nothing else.  You should see many console messages as proof of work.")
print("stellavslam module content:", dir(vslam))
config = vslam.config(config_file_path="./equirectangular.yaml")
SLAM = vslam.system(cfg=config, vocab_file_path="./orb_vocab.fbow")
SLAM.startup()
SLAM.shutdown()
print('Success!')