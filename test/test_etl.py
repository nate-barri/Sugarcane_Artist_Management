import psycopg2
import pandas as pd
import emoji
from psycopg2.extras import execute_values
from datetime import datetime

# Database connection parameters
db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_OTh3sXnBH5uL',
    'host': 'ep-lingering-sky-a1pnaim3-pooler.ap-southeast-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

def test_connection():
    """Test database connection"""
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print("✅ Connection successful!")
        print(f"PostgreSQL version: {version[0][:60]}...")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def setup_complete_schema():
    """Create entire data warehouse schema and sync function"""
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        print("\n📦 Setting up complete schema...")
        
        # Create schema
        cur.execute("CREATE SCHEMA IF NOT EXISTS dw;")
        print("   ✓ Schema created")
        
        # Platform dimension
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dw.dim_platform (
              platform_sk   smallserial PRIMARY KEY,
              platform_code text UNIQUE NOT NULL,
              platform_name text NOT NULL
            );
        """)
        cur.execute("""
            INSERT INTO dw.dim_platform(platform_code, platform_name)
            VALUES ('facebook','Facebook'), ('youtube','YouTube'), ('tiktok','TikTok')
            ON CONFLICT (platform_code) DO NOTHING;
        """)
        print("   ✓ Platform dimension created")
        
        # Date dimension
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dw.dim_date (
              date_sk          integer PRIMARY KEY,
              date_actual      date    NOT NULL,
              year             integer NOT NULL,
              month            integer NOT NULL,
              day              integer NOT NULL,
              dow_iso          integer NOT NULL,
              week_iso         integer NOT NULL,
              month_name       text    NOT NULL,
              quarter          integer NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_dim_date_date ON dw.dim_date(date_actual);
        """)
        print("   ✓ Date dimension created")
        
        # Date filler function
        cur.execute("""
            CREATE OR REPLACE FUNCTION dw.fill_dim_date(p_start date, p_end date)
            RETURNS void
            LANGUAGE plpgsql
            AS $$
            DECLARE d date;
            BEGIN
              IF p_start IS NULL THEN RETURN; END IF;
              IF p_end IS NULL THEN p_end := p_start; END IF;
              d := LEAST(p_start, p_end);
              WHILE d <= GREATEST(p_start, p_end) LOOP
                INSERT INTO dw.dim_date(date_sk, date_actual, year, month, day, dow_iso, week_iso, month_name, quarter)
                VALUES (
                  (to_char(d,'YYYYMMDD'))::int,
                  d,
                  EXTRACT(YEAR FROM d)::int,
                  EXTRACT(MONTH FROM d)::int,
                  EXTRACT(DAY FROM d)::int,
                  EXTRACT(ISODOW FROM d)::int,
                  EXTRACT(WEEK FROM d)::int,
                  to_char(d,'Mon'),
                  EXTRACT(QUARTER FROM d)::int
                )
                ON CONFLICT (date_sk) DO NOTHING;
                d := d + INTERVAL '1 day';
              END LOOP;
            END;
            $$;
        """)
        print("   ✓ Date filler function created")
        
        # Page dimension
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dw.dim_page (
              page_sk      bigserial PRIMARY KEY,
              platform_sk  smallint NOT NULL REFERENCES dw.dim_platform(platform_sk),
              page_id      text     NOT NULL,
              page_name    text,
              CONSTRAINT uq_dim_page UNIQUE (platform_sk, page_id)
            );
            CREATE INDEX IF NOT EXISTS ix_dim_page_id ON dw.dim_page(page_id);
        """)
        print("   ✓ Page dimension created")
        
        # Post dimension
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dw.dim_post (
              post_sk                bigserial PRIMARY KEY,
              platform_sk            smallint NOT NULL REFERENCES dw.dim_platform(platform_sk),
              page_sk                bigint   NOT NULL REFERENCES dw.dim_page(page_sk),
              post_id                text     NOT NULL,
              title                  text,
              description            text,
              post_type              text,
              duration_sec           int,
              published_at           timestamptz,
              permalink              text,
              is_crosspost           boolean,
              is_share               boolean,
              funded_content_status  smallint,
              CONSTRAINT uq_dim_post UNIQUE (platform_sk, post_id)
            );
            CREATE INDEX IF NOT EXISTS ix_dim_post_id ON dw.dim_post(post_id);
            CREATE INDEX IF NOT EXISTS ix_dim_post_published ON dw.dim_post(published_at);
        """)
        print("   ✓ Post dimension created")
        
        # Fact table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dw.fact_facebook_post_summary (
              fact_sk                   bigserial PRIMARY KEY,
              post_sk                   bigint  NOT NULL UNIQUE REFERENCES dw.dim_post(post_sk),
              date_sk_posted            integer REFERENCES dw.dim_date(date_sk),
              reach                     bigint  CHECK (reach >= 0),
              impressions               bigint  CHECK (impressions >= 0),
              shares                    int     CHECK (shares >= 0),
              comments                  int     CHECK (comments >= 0),
              reactions                 int     CHECK (reactions >= 0),
              seconds_viewed            numeric(20,4) CHECK (seconds_viewed >= 0),
              average_seconds_viewed    numeric(12,4) CHECK (average_seconds_viewed >= 0),
              created_at                timestamptz NOT NULL DEFAULT now(),
              updated_at                timestamptz NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS ix_fb_fact_post ON dw.fact_facebook_post_summary(post_sk);
            CREATE INDEX IF NOT EXISTS ix_fb_fact_date ON dw.fact_facebook_post_summary(date_sk_posted);
        """)
        print("   ✓ Fact table created")
        
        # Landing table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS facebook_data_set(
                post_id NUMERIC PRIMARY KEY,
                page_id VARCHAR,
                page_name TEXT,
                title TEXT,
                description TEXT,
                post_type TEXT,
                duration_sec NUMERIC(10,2) DEFAULT 0,
                publish_time TIMESTAMP,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                time TEXT,
                permalink TEXT,
                is_crosspost BOOLEAN,
                is_share BOOLEAN,
                funded_content_status TEXT,
                reach NUMERIC(20,2),
                shares NUMERIC(20,2),
                comments NUMERIC(20,2),
                reactions NUMERIC(20,2),
                seconds_viewed NUMERIC(10,2),
                average_seconds_viewed NUMERIC(10,2) DEFAULT 0,
                impressions NUMERIC(20,2) DEFAULT NULL
            );
        """)
        print("   ✓ Landing table created")
        
        # Sync function
        cur.execute("""
            CREATE OR REPLACE FUNCTION dw.sync_facebook_from_landing()
            RETURNS void
            LANGUAGE plpgsql
            AS $$
            DECLARE
              v_platform_sk smallint;
              v_min_date date;
              v_max_date date;
            BEGIN
              SELECT platform_sk INTO v_platform_sk 
              FROM dw.dim_platform 
              WHERE platform_code = 'facebook';

              IF v_platform_sk IS NULL THEN
                RAISE EXCEPTION 'Facebook platform not found in dim_platform';
              END IF;

              SELECT 
                MIN(COALESCE(make_date(year, month, day), publish_time::date)),
                GREATEST(MAX(COALESCE(make_date(year, month, day), publish_time::date)), CURRENT_DATE)
              INTO v_min_date, v_max_date
              FROM public.facebook_data_set
              WHERE publish_time IS NOT NULL
                 OR (year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL);

              IF v_min_date IS NOT NULL THEN
                PERFORM dw.fill_dim_date(v_min_date, v_max_date);
              END IF;

              -- Upsert Pages
              WITH src AS (
                SELECT DISTINCT
                  v_platform_sk AS platform_sk,
                  NULLIF(TRIM(page_id::text), '') AS page_id,
                  page_name
                FROM public.facebook_data_set
                WHERE page_id IS NOT NULL
              ),
              dedup AS (
                SELECT platform_sk, page_id, page_name,
                  ROW_NUMBER() OVER (PARTITION BY platform_sk, page_id 
                                     ORDER BY CASE WHEN page_name IS NOT NULL THEN 0 ELSE 1 END, page_name) AS rn
                FROM src
              )
              INSERT INTO dw.dim_page(platform_sk, page_id, page_name)
              SELECT platform_sk, page_id, page_name FROM dedup WHERE rn = 1
              ON CONFLICT (platform_sk, page_id) DO UPDATE
              SET page_name = COALESCE(EXCLUDED.page_name, dw.dim_page.page_name);

              -- Upsert Posts
              WITH src AS (
                SELECT
                  v_platform_sk AS platform_sk,
                  p.page_sk,
                  f.post_id::text AS post_id,
                  f.title, f.description, f.post_type,
                  CASE WHEN f.duration_sec IS NOT NULL THEN ROUND(f.duration_sec)::int ELSE NULL END AS duration_sec,
                  COALESCE(
                    CASE WHEN f.year IS NOT NULL AND f.month IS NOT NULL AND f.day IS NOT NULL
                         THEN make_timestamp(f.year, f.month, f.day,
                                COALESCE(NULLIF(split_part(f.time::text, ':', 1), '')::int, 0),
                                COALESCE(NULLIF(split_part(f.time::text, ':', 2), '')::int, 0),
                                COALESCE(NULLIF(split_part(f.time::text, ':', 3), '')::int, 0))
                    END,
                    f.publish_time
                  )::timestamptz AS published_at,
                  f.permalink, f.is_crosspost, f.is_share,
                  CASE WHEN f.funded_content_status ~ '^\d+$' THEN f.funded_content_status::int ELSE NULL END AS funded_content_status,
                  f.reach, f.impressions
                FROM public.facebook_data_set f
                JOIN dw.dim_page p ON p.platform_sk = v_platform_sk AND p.page_id = f.page_id::text
                WHERE f.post_id IS NOT NULL AND TRIM(f.post_id::text) <> ''
              ),
              dedup AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY platform_sk, post_id 
                          ORDER BY COALESCE(reach, 0) DESC, COALESCE(impressions, 0) DESC, published_at DESC NULLS LAST) AS rn
                FROM src
              )
              INSERT INTO dw.dim_post(platform_sk, page_sk, post_id, title, description, post_type, 
                                      duration_sec, published_at, permalink, is_crosspost, is_share, funded_content_status)
              SELECT platform_sk, page_sk, post_id, title, description, post_type,
                     duration_sec, published_at, permalink, is_crosspost, is_share, funded_content_status
              FROM dedup WHERE rn = 1
              ON CONFLICT (platform_sk, post_id) DO UPDATE
              SET title = COALESCE(EXCLUDED.title, dw.dim_post.title),
                  description = COALESCE(EXCLUDED.description, dw.dim_post.description),
                  post_type = COALESCE(EXCLUDED.post_type, dw.dim_post.post_type),
                  duration_sec = COALESCE(EXCLUDED.duration_sec, dw.dim_post.duration_sec),
                  published_at = COALESCE(EXCLUDED.published_at, dw.dim_post.published_at),
                  permalink = COALESCE(EXCLUDED.permalink, dw.dim_post.permalink),
                  is_crosspost = COALESCE(EXCLUDED.is_crosspost, dw.dim_post.is_crosspost),
                  is_share = COALESCE(EXCLUDED.is_share, dw.dim_post.is_share),
                  funded_content_status = COALESCE(EXCLUDED.funded_content_status, dw.dim_post.funded_content_status);

              -- Upsert Fact
              WITH src AS (
                SELECT f.post_id::text AS post_id, f.reach, f.impressions, f.shares, f.comments, f.reactions,
                       f.seconds_viewed, f.average_seconds_viewed,
                       COALESCE(CASE WHEN f.year IS NOT NULL AND f.month IS NOT NULL AND f.day IS NOT NULL
                                     THEN make_date(f.year, f.month, f.day) END, f.publish_time::date) AS posted_date
                FROM public.facebook_data_set f
                WHERE f.post_id IS NOT NULL AND TRIM(f.post_id::text) <> ''
              ),
              dedup AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY COALESCE(reach, 0) DESC, COALESCE(impressions, 0) DESC) AS rn
                FROM src
              ),
              cleaned AS (
                SELECT post_id, posted_date,
                  CASE WHEN reach::text ~* '^(nan|inf|-inf)$' THEN NULL ELSE reach END AS reach_clean,
                  CASE WHEN impressions::text ~* '^(nan|inf|-inf)$' THEN NULL ELSE impressions END AS impressions_clean,
                  CASE WHEN shares::text ~* '^(nan|inf|-inf)$' THEN NULL ELSE shares END AS shares_clean,
                  CASE WHEN comments::text ~* '^(nan|inf|-inf)$' THEN NULL ELSE comments END AS comments_clean,
                  CASE WHEN reactions::text ~* '^(nan|inf|-inf)$' THEN NULL ELSE reactions END AS reactions_clean,
                  CASE WHEN seconds_viewed::text ~* '^(nan|inf|-inf)$' THEN NULL ELSE seconds_viewed END AS seconds_viewed_clean,
                  CASE WHEN average_seconds_viewed::text ~* '^(nan|inf|-inf)$' THEN NULL ELSE average_seconds_viewed END AS avg_seconds_viewed_clean
                FROM dedup WHERE rn = 1
              )
              INSERT INTO dw.fact_facebook_post_summary(
                post_sk, date_sk_posted, reach, impressions, shares, comments, reactions,
                seconds_viewed, average_seconds_viewed, updated_at
              )
              SELECT dp.post_sk, (to_char(c.posted_date, 'YYYYMMDD'))::int,
                COALESCE(c.reach_clean, 0)::bigint, COALESCE(c.impressions_clean, 0)::bigint,
                COALESCE(c.shares_clean, 0)::int, COALESCE(c.comments_clean, 0)::int,
                COALESCE(c.reactions_clean, 0)::int, COALESCE(c.seconds_viewed_clean, 0)::numeric(20, 4),
                COALESCE(c.avg_seconds_viewed_clean, 0)::numeric(12, 4), now()
              FROM cleaned c
              JOIN dw.dim_post dp ON dp.platform_sk = v_platform_sk AND dp.post_id = c.post_id
              ON CONFLICT (post_sk) DO UPDATE
              SET reach = EXCLUDED.reach, impressions = EXCLUDED.impressions,
                  shares = EXCLUDED.shares, comments = EXCLUDED.comments, reactions = EXCLUDED.reactions,
                  seconds_viewed = EXCLUDED.seconds_viewed, average_seconds_viewed = EXCLUDED.average_seconds_viewed,
                  date_sk_posted = EXCLUDED.date_sk_posted, updated_at = now();

              RAISE NOTICE 'Facebook sync completed successfully';
            EXCEPTION
              WHEN OTHERS THEN
                RAISE NOTICE 'Error in sync_facebook_from_landing: %', SQLERRM;
                RAISE;
            END;
            $$;
        """)
        print("   ✓ Sync function created")
        
        # Trigger function
        cur.execute("""
            CREATE OR REPLACE FUNCTION dw.trg_fb_sync_stmt()
            RETURNS trigger
            LANGUAGE plpgsql
            SECURITY DEFINER
            SET search_path = dw, public
            AS $$ 
            BEGIN
              PERFORM dw.sync_facebook_from_landing();
              RETURN NULL;
            END
            $$;
        """)
        print("   ✓ Trigger function created")
        
        # Create trigger
        cur.execute("DROP TRIGGER IF EXISTS fb_sync_stmt_insupd ON public.facebook_data_set;")
        cur.execute("""
            CREATE TRIGGER fb_sync_stmt_insupd
            AFTER INSERT OR UPDATE ON public.facebook_data_set
            FOR EACH STATEMENT
            EXECUTE FUNCTION dw.trg_fb_sync_stmt();
        """)
        print("   ✓ Trigger created")
        
        conn.commit()
        cur.close()
        conn.close()
        
        print("\n✅ Complete schema setup successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Schema setup failed: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False

def remove_emojis(text):
    """Remove emojis from text"""
    return emoji.replace_emoji(text, "") if isinstance(text, str) else text

def load_and_insert_csv(file_path):
    """Load CSV and insert into database"""
    conn = None
    try:
        print(f"\n📂 Loading CSV: {file_path}")
        
        # Read CSV
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        except:
            df = pd.read_csv(file_path, encoding='utf-8')
        
        print(f"   Found {len(df)} rows")
        
        # Clean column names
        df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9_]', '_', regex=True)
        df.columns = df.columns.str.replace(r'^[0-9]', 'col_', regex=True)
        
        # Replace NaNs
        df = df.where(pd.notna(df), None)
        
        # Remove emojis
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].apply(remove_emojis)
        
        # Convert publish_time
        if 'publish_time' in df.columns:
            df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
            df['year'] = df['publish_time'].dt.year
            df['month'] = df['publish_time'].dt.month
            df['day'] = df['publish_time'].dt.day
            df['time'] = df['publish_time'].dt.strftime("%H:%M:%S")
        
        # Convert numeric columns
        decimal_columns = ['duration_sec', 'seconds_viewed', 'average_seconds_viewed']
        for col in decimal_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
        
        # Convert boolean fields
        boolean_columns = ['is_crosspost', 'is_share']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Set impressions
        if 'post_type' in df.columns and 'reach' in df.columns:
            df['impressions'] = df.apply(lambda row: row['reach'] if row['post_type'] == 'photo' else None, axis=1)
        
        # Convert numeric columns
        numeric_columns = ['post_id', 'reach', 'shares', 'comments', 'reactions', 'impressions']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicates
        df = df.sort_values(by=['reach', 'impressions'], ascending=False, na_position='last')
        df = df.drop_duplicates(subset=['post_id'], keep='first')
        
        print(f"   After deduplication: {len(df)} rows")
        
        # Insert into database
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Get table columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'facebook_data_set'
        """)
        table_columns = {row[0] for row in cur.fetchall()}
        
        # Keep only matching columns
        df = df[[col for col in df.columns if col in table_columns]]
        
        columns = ', '.join(df.columns)
        update_columns = [col for col in df.columns if col != 'post_id']
        update_set = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_columns])
        
        sql = f"""
        INSERT INTO facebook_data_set ({columns}) 
        VALUES %s
        ON CONFLICT (post_id) DO UPDATE
        SET {update_set};
        """
        
        values = [tuple(row) for row in df.values]
        execute_values(cur, sql, values)
        conn.commit()
        
        cur.execute("SELECT COUNT(*) FROM facebook_data_set")
        total = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        print(f"✅ Data inserted! Total records in database: {total}")
        return True
        
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False

def verify_data_warehouse():
    """Verify data in the warehouse"""
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        print("\n🔍 Verifying Data Warehouse...")
        
        cur.execute("SELECT COUNT(*) FROM facebook_data_set")
        landing = cur.fetchone()[0]
        print(f"   Landing table: {landing:,} records")
        
        cur.execute("SELECT COUNT(*) FROM dw.dim_page")
        pages = cur.fetchone()[0]
        print(f"   Pages: {pages:,} records")
        
        cur.execute("SELECT COUNT(*) FROM dw.dim_post")
        posts = cur.fetchone()[0]
        print(f"   Posts: {posts:,} records")
        
        cur.execute("SELECT COUNT(*) FROM dw.fact_facebook_post_summary")
        facts = cur.fetchone()[0]
        print(f"   Facts: {facts:,} records")
        
        if facts > 0:
            print("\n📊 Top 5 Posts by Engagement:")
            cur.execute("""
                SELECT 
                    p.post_id,
                    LEFT(p.title, 40) as title,
                    p.post_type,
                    f.reach,
                    f.reactions,
                    f.comments,
                    f.shares
                FROM dw.fact_facebook_post_summary f
                JOIN dw.dim_post p ON p.post_sk = f.post_sk
                ORDER BY f.reach DESC NULLS LAST
                LIMIT 5
            """)
            for row in cur.fetchall():
                print(f"   ID: {row[0]} | {row[1]}... | Type: {row[2]} | Reach: {row[3]:,} | ❤️ {row[4]:,} | 💬 {row[5]:,} | 🔄 {row[6]:,}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        if conn:
            conn.close()
        return False

def main():
    """Main execution"""
    print("=" * 70)
    print("FACEBOOK DATA WAREHOUSE - COMPLETE SETUP")
    print("=" * 70)
    
    # Step 1: Test connection
    if not test_connection():
        return
    
    # Step 2: Setup schema
    if not setup_complete_schema():
        return
    
    # Step 3: Load CSV (if file exists)
    csv_file = r"C:\Sugarcane_Artist_Management\test\FULL_SET_FB.csv"
    import os
    if os.path.exists(csv_file):
        if load_and_insert_csv(csv_file):
            # Step 4: Verify
            verify_data_warehouse()
    else:
        print(f"\n⚠️ CSV file not found: {csv_file}")
        print("   Please update the file path and run again")
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print("\nYour data warehouse is ready!")
    print("• Trigger is active - new data will auto-sync")
    print("• Use dw.vw_fb_top_posts view for analytics")
    print("• Run verify_data_warehouse() to check data anytime")

if __name__ == "__main__":
    main()