import fcntl, struct, sys, time, os

UI_DEV_CREATE, UI_DEV_DESTROY = 0x5501, 0x5502
UI_SET_EVBIT, UI_SET_KEYBIT = 0x40045564, 0x40045565
EV_SYN, EV_KEY, SYN_REPORT = 0, 1, 0

KEYS = {'a':30,'b':48,'c':46,'d':32,'e':18,'f':33,'g':34,'h':35,'i':23,'j':36,
        'k':37,'l':38,'m':50,'n':49,'o':24,'p':25,'q':16,'r':19,'s':31,'t':20,
        'u':22,'v':47,'w':17,'x':45,'y':21,'z':44,' ':57}
SUPER, ESC, BACKSPACE = 125, 1, 14

fd = os.open('/dev/uinput', os.O_WRONLY | os.O_NONBLOCK)
fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY)
for code in list(KEYS.values()) + [SUPER, ESC, BACKSPACE]:
    fcntl.ioctl(fd, UI_SET_KEYBIT, code)
# struct uinput_user_dev: name[80], input_id(4*u16), ff_effects_max, 4*64 ints
dev = b'virglrs-corpus-kbd'.ljust(80, b'\0') + struct.pack('HHHH', 0x03, 0x1234, 0x5678, 1) \
      + struct.pack('i', 0) + b'\0' * (64 * 4 * 4)
os.write(fd, dev)
fcntl.ioctl(fd, UI_DEV_CREATE)
time.sleep(2)   # let mutter/libinput adopt it

def ev(t, c, v):
    os.write(fd, struct.pack('qqHHi', 0, 0, t, c, v))

def tap(code, hold=0.03):
    ev(EV_KEY, code, 1); ev(EV_SYN, SYN_REPORT, 0); time.sleep(hold)
    ev(EV_KEY, code, 0); ev(EV_SYN, SYN_REPORT, 0); time.sleep(0.06)

for word in sys.argv[1:]:
    tap(SUPER); time.sleep(1.2)                 # overview
    for ch in word:
        tap(KEYS[ch]); time.sleep(0.12)         # each keystroke lays out new text
    time.sleep(1.5)
    for _ in word:
        tap(BACKSPACE); time.sleep(0.08)
    tap(ESC); time.sleep(0.8)

fcntl.ioctl(fd, UI_DEV_DESTROY)
os.close(fd)
print("typed", " ".join(sys.argv[1:]))
