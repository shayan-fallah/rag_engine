-- A sample products table, like the one in your shop's MySQL database.
CREATE TABLE IF NOT EXISTS products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    brand VARCHAR(100),
    price DECIMAL(10, 2),
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARSET = utf8mb4;

INSERT INTO products (name, description, category, brand, price, is_active) VALUES
('Galaxy S24', '<p>Samsung flagship phone with a <b>50MP camera</b>, 8GB RAM and a 6.2" display. Model SM-S921B.</p>', 'Phones', 'Samsung', 799.00, 1),
('Pixel 9', 'Google phone with an excellent camera, seven years of OS updates and Gemini built in.', 'Phones', 'Google', 699.00, 1),
('Redmi Note 13', 'Budget phone with a 108MP camera and a 5000mAh battery that lasts two days.', 'Phones', 'Xiaomi', 229.00, 1),
('QuietComfort Ultra', 'Wireless noise cancelling headphones with 24 hours of battery and spatial audio.', 'Audio', 'Bose', 429.00, 1),
('WH-1000XM5', 'Sony wireless headphones with industry-leading noise cancelling and 30 hour battery.', 'Audio', 'Sony', 399.00, 1),
('AirPods Pro 2', 'Apple earbuds with active noise cancellation, transparency mode and USB-C charging case.', 'Audio', 'Apple', 249.00, 1),
('Speedgoat 6', 'Lightweight trail running shoes with a grippy Vibram outsole for mountain trails.', 'Shoes', 'Hoka', 155.00, 1),
('Pegasus 41', 'Everyday road running shoes with responsive cushioning, good for beginners.', 'Shoes', 'Nike', 140.00, 1),
('MacBook Air 13 M3', 'Thin and light laptop with 18 hours of battery life, 16GB memory and a fanless design.', 'Laptops', 'Apple', 1099.00, 1),
('ThinkPad X1 Carbon', 'Business laptop with a great keyboard, 14" screen and Intel Core Ultra processor.', 'Laptops', 'Lenovo', 1499.00, 1),
('گوشی موبایل گلکسی A55', 'گوشی میان‌رده سامسونگ با دوربین ۵۰ مگاپیکسلی، باتری ۵۰۰۰ میلی‌آمپر و بدنه ضد آب.', 'Phones', 'Samsung', 449.00, 1),
('Old model phone', 'Discontinued phone that is no longer sold.', 'Phones', 'Nokia', 99.00, 0);
