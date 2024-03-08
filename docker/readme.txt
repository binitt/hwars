#Updating chrome (it kept on crashing even after upgrade)

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 78BD65473CB3BD13
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 78BD65473CB3BD13

apt-get update
apt-get --only-upgrade install google-chrome-stable


#Updating firefox
apt install --only-upgrade firefox


touch ~/.Xauthority

apt-get install python3-tk python3-dev
