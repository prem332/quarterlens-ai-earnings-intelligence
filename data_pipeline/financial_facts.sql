IF OBJECT_ID('dbo.financial_facts', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.financial_facts (
        id            INT IDENTITY(1,1) NOT NULL,
        ticker        NVARCHAR(10)   NOT NULL,
        cik           NVARCHAR(10)   NOT NULL,   -- zero-padded, matches manifest
        accession     NVARCHAR(25)   NOT NULL,   -- e.g. 0000320193-24-000123
        form          NVARCHAR(10)   NOT NULL,   -- 10-Q / 10-K
        fiscal_label  NVARCHAR(16)   NOT NULL,   -- e.g. Q2_2025
        concept       NVARCHAR(128)  NOT NULL,   -- canonical name (e.g. Revenues)
        xbrl_tag      NVARCHAR(128)  NOT NULL,   -- actual us-gaap tag that resolved
        value         DECIMAL(28,4)  NOT NULL,   -- $B-scale and fractional EPS share one column
        unit          NVARCHAR(20)   NOT NULL,   -- USD, USD/shares, shares
        period_start  DATE           NULL,       -- NULL for instant (balance-sheet) facts
        period_end    DATE           NOT NULL,
        fy            SMALLINT       NULL,        -- XBRL fiscal year
        fp            NVARCHAR(4)    NULL,        -- XBRL fiscal period (Q1..Q4, FY)
        loaded_at     DATETIME2(0)   NOT NULL
                      CONSTRAINT DF_financial_facts_loaded_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_financial_facts PRIMARY KEY (id),
        CONSTRAINT UQ_financial_facts UNIQUE (accession, concept, period_start, period_end)
    );
END
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_financial_facts_lookup'
      AND object_id = OBJECT_ID('dbo.financial_facts')
)
    CREATE NONCLUSTERED INDEX IX_financial_facts_lookup
        ON dbo.financial_facts (cik, fiscal_label, concept)
        INCLUDE (value, unit, period_start, period_end);
GO