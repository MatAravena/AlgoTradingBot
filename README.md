# AlgoTradingBot

Trading Bot 

Strategy RSI and MACD
Broker Alpaca
Container AWS EC2


## EC2 Setup

1. Launch EC2 instance
2. Install Docker:
```bash
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

## docker build

docker build -t algobot .
docker run -d --env-file .env algobot