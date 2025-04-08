CREATE TABLE products (
    gtin            CHAR(13),
    product_code    VARCHAR(15),
    size            VARCHAR(10),
    color           VARCHAR(30),
    label           VARCHAR(100),
    category        VARCHAR(50),

    PRIMARY KEY (gtin, product_code)
);

CREATE TABLE sales (
    sku         VARCHAR(15),
    quantity    SMALLINT,
    site_code   VARCHAR(8),
    date        DATE
);

CREATE TABLE soh (
    site_code   VARCHAR(8),
    sku         VARCHAR(15),
    quantity    INTEGER,
    date        DATE, 

    PRIMARY KEY (site_code, sku, date) -- composed pk
);
CREATE INDEX idx_sales_sku_site_code ON sales (sku, site_code);
CREATE INDEX idx_soh_site_sku_date ON soh (site_code, sku, date);
CREATE INDEX idx_products_category ON products (category);

DROP MATERIALIZED VIEW IF EXISTS soh_batch_3_months;
CREATE MATERIALIZED VIEW soh_batch_3_months as 
SELECT 
    sku,
    site_code,
    DATE_TRUNC('quarter', date) AS batch_date,
    quantity
FROM (
    SELECT 
        sku,
        site_code,
        date,
        quantity,
        ROW_NUMBER() OVER (
            PARTITION BY sku, site_code, DATE_TRUNC('quarter', date)
            ORDER BY date DESC
        ) AS row_number
    FROM soh
) subquery
WHERE row_number = 1;


SELECT 
    p.category, 
    SUM(s.quantity) as quantity, 
    s.date 
FROM
    soh s 
INNER JOIN 
    products p ON p.product_code = s.sku
WHERE s.site_code = ''
    AND date in (
            SELECT MAX(date) 
            FROM soh 
            WHERE site_code = 'USA001' 
        ) 
    AND p.category = ''
GROUP BY p.category, s.date;