Testing:

for i in {1..183}; do python3 run.py -m movidius -ms generate -tf tests/testconfig.txt -ti ${i} -s settings.json; sudo ./usbreset $(lsusb | grep 03e7 | awk '{print "/dev/bus/usb/"$2"/"$4}' | sed 's/.$//'); done

The number 250 is the number of tests in the testconfig.txt file.
The string 03e7 is the first characters of the USB device's hardware ID.