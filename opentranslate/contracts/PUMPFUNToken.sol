// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title PUMPFUN Token
 * @dev ERC20 Token for the OpenTranslate platform
 */
contract PUMPFUNToken is ERC20, ERC20Burnable, Pausable, Ownable {
    uint256 public constant INITIAL_SUPPLY = 1_000_000_000 * 10**18; // 1 billion tokens
    uint256 public constant BURN_RATE = 1; // 1% burn rate
    
    mapping(address => bool) public minters;
    
    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);
    event TokensBurned(address indexed burner, uint256 amount);
    
    constructor() ERC20("PUMPFUN Token", "PUMPFUN") {
        _mint(msg.sender, INITIAL_SUPPLY);
    }
    
    modifier onlyMinter() {
        require(minters[msg.sender], "Caller is not a minter");
        _;
    }
    
    function addMinter(address _minter) external onlyOwner {
        require(_minter != address(0), "Invalid minter address");
        minters[_minter] = true;
        emit MinterAdded(_minter);
    }
    
    function removeMinter(address _minter) external onlyOwner {
        minters[_minter] = false;
        emit MinterRemoved(_minter);
    }
    
    function mint(address _to, uint256 _amount) external onlyMinter {
        _mint(_to, _amount);
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }
    
    function _afterTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override {
        super._afterTokenTransfer(from, to, amount);
        
        // Apply burn rate for non-excluded transfers
        if (from != address(0) && to != address(0)) {
            uint256 burnAmount = (amount * BURN_RATE) / 100;
            if (burnAmount > 0) {
                _burn(to, burnAmount);
                emit TokensBurned(to, burnAmount);
            }
        }
    }
    
    function decimals() public pure override returns (uint8) {
        return 18;
    }
} 