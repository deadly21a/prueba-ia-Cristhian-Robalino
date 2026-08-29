-- PostgreSQL function requested by the technical assessment.
-- SQLite is used for local development; this function is ready for PostgreSQL deployment.
CREATE OR REPLACE FUNCTION get_active_customer_ticket_count(p_customer_id INTEGER)
RETURNS INTEGER AS $$
    SELECT COUNT(*)::INTEGER
    FROM tickets
    WHERE customer_id = p_customer_id AND is_active = TRUE;
$$ LANGUAGE SQL STABLE;
