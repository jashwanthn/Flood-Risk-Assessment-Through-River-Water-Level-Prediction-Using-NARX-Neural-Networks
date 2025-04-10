-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 15, 2024 at 12:44 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `carbonfootprint_2024`
--

-- --------------------------------------------------------

--
-- Table structure for table `predictdata`
--

CREATE TABLE `predictdata` (
  `id` int(10) NOT NULL,
  `SUBDIVISION` varchar(255) DEFAULT NULL,
  `YEAR` varchar(255) DEFAULT NULL,
  `JAN` varchar(255) DEFAULT NULL,
  `FEB` varchar(255) DEFAULT NULL,
  `MAR` varchar(255) DEFAULT NULL,
  `APR` varchar(255) DEFAULT NULL,
  `MAY` varchar(255) DEFAULT NULL,
  `JUN` varchar(255) DEFAULT NULL,
  `JUL` varchar(255) DEFAULT NULL,
  `AUG` varchar(255) DEFAULT NULL,
  `SEP` varchar(255) DEFAULT NULL,
  `OCT` varchar(255) DEFAULT NULL,
  `NOV` varchar(255) DEFAULT NULL,
  `DEC` varchar(255) DEFAULT NULL,
  `ANNUAL RAINFALL` varchar(255) DEFAULT NULL,
  `FLOODS` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

--
-- Dumping data for table `predictdata`
--

INSERT INTO `predictdata` (`id`, `SUBDIVISION`, `YEAR`, `JAN`, `FEB`, `MAR`, `APR`, `MAY`, `JUN`, `JUL`,`AUG`,`SEP`,`OCT`,`NOV`,`DEC`,`ANNUAL RAINFALL`,`FLOODS`) VALUES
(1, 'KERALA','1901','28.7','44.7','51.6','160','174.7','824.6','743','357.5','197.7','266.9','350.8','48.4','3248.6','YES');


-- --------------------------------------------------------

--
-- Table structure for table `userdata`
--

CREATE TABLE `userdata` (
  `Named` varchar(50) DEFAULT NULL,
  `Email` varchar(50) DEFAULT NULL,
  `Pswd` varchar(50) DEFAULT NULL,
  `Phone` varchar(50) DEFAULT NULL,
  `Addr` varchar(4000) DEFAULT NULL,
  `Dob` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `userdata`
--

INSERT INTO `userdata` (`Named`, `Email`, `Pswd`, `Phone`, `Addr`, `Dob`) VALUES
('Sunny Boyka', 'madhsunil@gmail.com', 'qqq', '9036453696', 'Mysore\njj', '12-12-1978'),
('user', 'user@user.com', 'user', '9876543210', 'address', '04/20/2024');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `predictdata`
--
ALTER TABLE `predictdata`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `predictdata`
--
ALTER TABLE `predictdata`
  MODIFY `id` int(10) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=60281;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
