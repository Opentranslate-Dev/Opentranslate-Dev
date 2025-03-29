const hre = require("hardhat");

async function main() {
  console.log("Starting deployment...");

  // Deploy PUMPFUN Token
  const PUMPFUNToken = await hre.ethers.getContractFactory("PUMPFUNToken");
  console.log("Deploying PUMPFUN Token...");
  const pumpfunToken = await PUMPFUNToken.deploy();
  await pumpfunToken.deployed();
  console.log("PUMPFUN Token deployed to:", pumpfunToken.address);

  // Deploy Translation Contract
  const Translation = await hre.ethers.getContractFactory("Translation");
  console.log("Deploying Translation Contract...");
  const translation = await Translation.deploy(pumpfunToken.address);
  await translation.deployed();
  console.log("Translation Contract deployed to:", translation.address);

  // Add Translation Contract as minter
  console.log("Adding Translation Contract as minter...");
  const addMinterTx = await pumpfunToken.addMinter(translation.address);
  await addMinterTx.wait();
  console.log("Translation Contract added as minter");

  // Verify contracts on Etherscan
  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    console.log("Waiting for block confirmations...");
    await pumpfunToken.deployTransaction.wait(6);
    await translation.deployTransaction.wait(6);

    console.log("Verifying contracts on Etherscan...");
    
    await hre.run("verify:verify", {
      address: pumpfunToken.address,
      constructorArguments: []
    });

    await hre.run("verify:verify", {
      address: translation.address,
      constructorArguments: [pumpfunToken.address]
    });
  }

  console.log("Deployment completed!");
  
  // Return the deployed contract addresses
  return {
    pumpfunToken: pumpfunToken.address,
    translation: translation.address
  };
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 