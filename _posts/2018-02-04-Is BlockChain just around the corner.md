---
layout:     post
title:      What is Blockchain
subtitle:   Is Blockchain application just around the corner?
date:       2018-02-04
author:     Jiayi
header-img: img/post-bg-ios10.jpg
catalog: true
tags:
    - BlockChain
---

Although the Internet has connected the world closer, it is accompanied with growing gap of trust. Looking at the current business environment, trust is probably the most we lack. Trust is actually a risk assessment, sometimes predictable. We will check each other's credit when trading in Ebay or Taobao. We pay and receive payment through Paypal or Alipay because we trust these intermediaries. But what if the servers of these service providers are down or even hacked? When dealing with a $10 breach , will you spend time on defending it? For small and medium enterprises, building trust is even harder.
Blockchain technology is designed in order to eliminate the barriers of trust. It is capable of verifying who is on the other side of the transaction in an effective way, and help people farewell to the tedious process of confirming trust.


##Blockchain 1.0##

Blockchain involves peer to peer network, distributed storage, cryptography, game theory and other cross-cutting technologies.
The first application of Blockchain - Bitcoin, the first digital currency to solve the double spending problem without requiring a trusted authority -  is called Blockchain 1.0, it has the following characteristics:
1. Bitcoin is an open distributed ledger
Anyone on the node can get a copy of the ledger, even can record transaction. But that does not mean that anyone knows how much money they have, and transaction recording does not mean people can tamper with the data at will.

2. De-centralization
By encryption and signature via "Asymmetric Encryption" technology, people exercises the right of his/her account (public key) without revealing the password (private key). This process does not require traditional intermediaries such as banking systems.

3. Non-repudiation and traceable
When you record transaction in a new page of the ledger, it compresses all the information of the old page into a string of characters (HASH value) and writes it to the beginning of the new page. Because of the time stamp, any tampering with any previous page information will result in the inconsistency of the new page. This page is called a Block, each page is linked by the page number and time stamp into a full ledger, called Blockchain.
All those who want to record transaction must calculate a math problem. Based on the number of the previous page and the new transfer instruction to be recorded, the person who first answers the question has the right to record transaction. In another eight minutes, the system then proceeds with a new question based on the new transfer order.

4. 51% attack
Because transaction recording is a global race for all miners, the premise of successful cheating is that cheater controls the 51% mining power (i.e. computing power) more than all other miners.

Blockchain 2.0 

Blockchain 2.0 is a combination of digital currency and smart contracts. The application is Ethereum which is more than just a cryptocurrency for trading: its real value is to make people to use distribution computers supported by thousands of nodes globally.

Smart contract
Ethereum needs thousands of people to run software on their computers to support the entire network. Each node (computer) in the network runs the Ethereum Virtual Machine (EVM). Imagine EVM as an operating system, and the software or application executed by EVM is called a "Smart Contract." Smart contract can be partially or fully executed or enforced without human interaction (i.e. automated escrow).  A Blockchain smart contract would be enabled by extensible programming instructions that define and execute an agreement.

For example, suppose you and I bet Super Bowl. I bet Eagles wins 2018 Super Bowl, but you bet Patriots wins. We agree that the loser must pay $100 to the winner. We may write a contract on Ethereum that will receive $100 from both of us. By the end of the game we will know who is the champion of Super Bowl through an open API in NBC, and the smart contract will send $200 to the winning party.

If a user wants to use and execute a smart contract, he or she must pay the execution fee, which may be calculated based on the nodes that actually spend resources such as memory, storage, computing power to execute the contract. Each statement in the contract has a fee. For example, if the statement executed uses the memory of the node, this statement will have a price. If you execute a statement using the node's hard disk storage, these statements are also priced.

Summary:

Although there are still many places where the underlying technologies should be improved, the platforms of Blockchain are semi-mature and can be commercially launched in the near future. The commercialization is the biggest challenge.

Obstacles:

1. Too few types of consensus mechanisms (proof of work, proof of stake, etc) to meet diversified business scenarios
2. Scalability: Small capacity of distributed storage block, leading to network congestion. It is unable to meet data volume requirements in banks, stock exchanges. As of late 2016, Blockchain can only process about seven transactions per second, and each transaction costs about $0.20 and can only store 80 bytes of data.
3 Network size:  This requires a large network of users, however. If a Blockchain is not a robust network with a widely distributed grid of nodes, it becomes more difficult to reap the full benefit.
4. The database is different from the traditional database. Blockchain applications require extensive writes, HASH calculations and validation.
5. A lack of consumer "friendliness"
Read more challenges in Forbes

Opportunities:

Is Blockchain 3.0 just around the corner? It is expected to address a) the underlying ecosystem of the entire platform, b) the common application of Blockchain, as well as c) the applications in various vertical industries.

a. The underlying ecosystem
The ecosystem maintains the high performance of commercial applications by de-centralization. It includes:
- Basic protocol: it is a complete Blockchain product, similar to a computer operating system to maintain network nodes, provide API calls, etc. Use network programming, distributed algorithms, encryption and signature, data storage and other technologies to build the network environment, transaction channels, node reward.
- Anonymous technology: Anonymous currency is implemented so that people know neither the relationship between the trader and the physical entity, nor the details of the transaction (amount, transaction time, sending receiver information). Use of coinjoin, blind signature, hidden address, ring signature, zero knowledge proof and other technologies.
- Blockchain Hardware: Composed of Bitcoin mining machine vendors and Blockchain router providers. Mining machine provides the computing power. Blockchain router providers use idle bandwidth for mining services.

b. the common application of Blockchain: it covers the service such as Smart Contracts, Mining Services, Data Services, Information Security, Enterprise Solutions, etc.

c. the applications in various vertical industries: Finance, Entertainment, Medical, Intelligent IoT Applications, Supply Chain Automation Management, etc.
