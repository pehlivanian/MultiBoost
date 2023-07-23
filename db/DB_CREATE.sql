/* Run as root: $sudo mariadb -u root -p < DB_CREATE.sql */

/* Drop databases */
DROP DATABASE IF EXISTS MULTISCALEGB_CLASS;
DROP DATABASE IF EXISTS MULTISCALEGB_REG;
DROP DATABASE IF EXISTS MULTISCALEGB_CLASS_SUMM;
DROP DATABASE IF EXISTS MULTISCALEGB_REG_SUMM;

/* Create databases */
CREATE DATABASE IF NOT EXISTS MULTISCALEGB_CLASS;
CREATE DATABASE IF NOT EXISTS MULTISCALEGB_REG;
CREATE DATABASE IF NOT EXISTS MULTISCALEGB_CLASS_SUMM;
CREATE DATABASE IF NOT EXISTS MULTISCALEGB_REG_SUMM;

/* Handling std::vector<T> input as std::array<T,10> input
as sql is crippled */
USE MULTISCALEGB_CLASS;
CREATE TABLE run_specification (
run_key CHAR(100),
folder CHAR(200),
idx CHAR(100),
dataset_name CHAR(100),
loss_fn INT,
n_rows INT,
n_cols INT,
basesteps INT,
colsubsample_ratio FLOAT(24),
recursive BOOLEAN,
split_ratio FLOAT(24),
num_partitions0 INT,
num_partitions1 INT,
num_partitions2 INT,
num_partitions3 INT,
num_partitions4 INT,
num_partitions5 INT,
num_partitions6 INT,
num_partitions7 INT,
num_partitions8 INT,
num_partitions9 INT,
num_steps0 INT,
num_steps1 INT,
num_steps2 INT,
num_steps3 INT,
num_steps4 INT,
num_steps5 INT,
num_steps6 INT,
num_steps7 INT,
num_steps8 INT,
num_steps9 INT,
learning_rate0 FLOAT(24),
learning_rate1 FLOAT(24),
learning_rate2 FLOAT(24),
learning_rate3 FLOAT(24),
learning_rate4 FLOAT(24),
learning_rate5 FLOAT(24),
learning_rate6 FLOAT(24),
learning_rate7 FLOAT(24),
learning_rate8 FLOAT(24),
learning_rate9 FLOAT(24),
max_depth0 INT,
max_depth1 INT,
max_depth2 INT,
max_depth3 INT,
max_depth4 INT,
max_depth5 INT,
max_depth6 INT,
max_depth7 INT,
max_depth8 INT,
max_depth9 INT,
min_leafsize0 INT,
min_leafsize1 INT,
min_leafsize2 INT,
min_leafsize3 INT,
min_leafsize4 INT,
min_leafsize5 INT,
min_leafsize6 INT,
min_leafsize7 INT,
min_leafsize8 INT,
min_leafsize9 INT,
min_gainsplit0 FLOAT(24),
min_gainsplit1 FLOAT(24),
min_gainsplit2 FLOAT(24),
min_gainsplit3 FLOAT(24),
min_gainsplit4 FLOAT(24),
min_gainsplit5 FLOAT(24),
min_gainsplit6 FLOAT(24),
min_gainsplit7 FLOAT(24),
min_gainsplit8 FLOAT(24),
min_gainsplit9 FLOAT(24));

CREATE TABLE insample (
run_key CHAR(100),
dataset_name CHAR(100),
iteration INT,
err FLOAT(24),
prcsn FLOAT(24),
recall FLOAT(24),
F1 FLOAT(24));

CREATE TABLE outofsample (
run_key CHAR(100),
dataset_name CHAR(100),
iteration INT,
err FLOAT(24),
prcsn FLOAT(24),
recall FLOAT(24),
F1 FLOAT(24)
);


/* Create user */
CREATE USER IF NOT EXISTS 'charles'@'localhost' IDENTIFIED BY 'gongzuo';
GRANT ALL PRIVILEGES ON MULTISCALEGB.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON MULTISCALEGB_CLASS.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON MULTISCALEGB_REG.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON MULTISCALEGB_CLASS_SUMM.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON MULTISCALEGB_REG_SUMM.* TO 'charles'@'localhost' IDENTIFIED BY 'gongzuo' WITH GRANT OPTION;


FLUSH PRIVILEGES;

