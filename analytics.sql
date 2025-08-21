CREATE TABLE analytics (
    post_id INT PRIMARY KEY,
    page_id BIGINT,  -- Changed to BIGINT to prevent integer overflow
    page_name TEXT,
    title TEXT,
    description TEXT,
    duration INT,
    publish_year INT,  -- Extracted year
    publish_month INT, -- Extracted month
    publish_day INT,   -- Extracted day
    publish_time TIME, -- Extracted time
    caption_type TEXT,
    permalink TEXT,
    is_crosspost BOOLEAN,
    is_share BOOLEAN,
    post_type TEXT,
    languages TEXT,
    custom_labels TEXT,
    funded_content_status TEXT,
    data_comment TEXT,
    date VARCHAR(90),
    impressions INT,
    reach INT,
    reactions_comments_shares INT,
    reactions INT,
    comments INT,
    shares INT,
    total_clicks INT,
    other_clicks INT,
    matched_audience_targeting_consumption INT,
    link_clicks INT,
    views INT,
    reels_plays_count INT,
    seconds_viewed INT,
    avg_seconds_viewed FLOAT,
    estimated_earnings FLOAT
);




truncate table analytics
drop table if exists analytics
select *from analytics