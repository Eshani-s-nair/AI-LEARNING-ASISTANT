import psycopg2
import os
from werkzeug.security import generate_password_hash, check_password_hash

class DatabaseManager:
    def __init__(self):
        self.config = {
            "dbname": "study_assistant",
            "user": "postgres",
            "password": "Postgres123!",  
            "host": "localhost",
            "port": "5432"
        }

    def save_data(self, file_path, summary_sentences, keywords, user_id=None, title=None):
        conn = None
        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()

            # 1. Insert document record
            file_name = os.path.basename(file_path)
            file_type = os.path.splitext(file_name)[1] or None
            cur.execute(
                "INSERT INTO documents (user_id, file_name, file_type, uploaded_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP) RETURNING id",
                (int(user_id) if user_id else None, file_name, file_type)
            )
            doc_id = cur.fetchone()[0]

            # 2. Insert summary
            summary_text = " ".join(summary_sentences)
            cur.execute(
                "INSERT INTO summaries (document_id, summary_text) VALUES (%s, %s)",
                (doc_id, summary_text)
            )

            # 3. Insert keywords
            if keywords:
                keyword_records = [(doc_id, kw[0]) for kw in keywords]
                cur.executemany(
                    "INSERT INTO keywords (document_id, keyword) VALUES (%s, %s)",
                    keyword_records
                )

            conn.commit()
            cur.close()
            print("\n" + "="*50)
            print(f"✅ Data saved to database successfully (Document ID: {doc_id})")
            print("="*50 + "\n")
            return doc_id

        except Exception as e:
            if conn:
                conn.rollback()
            print("\n" + "!"*40)
            print(f"⚠️ DATABASE ERROR: {e}")
            print("💡 Transaction rolled back. Data was NOT saved.")
            print("!"*40 + "\n")
            return None
        finally:
            if conn:
                conn.close()

    def create_user(self, first_name, last_name, email, username, password):
        """Creates a new user in the database with a hashed password."""
        conn = None
        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()

            # Check for existing user by username or email
            cur.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
            if cur.fetchone():
                cur.close()
                return {"success": False, "message": "Username or email already exists"}

            password_hash = generate_password_hash(password)

            cur.execute(
                """
                INSERT INTO users (first_name, last_name, email, username, password_hash)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (first_name, last_name, email, username, password_hash)
            )
            user_id = cur.fetchone()[0]

            conn.commit()
            cur.close()
            # Return user data for immediate login after signup
            return {"success": True, "user": {
                "id": user_id,
                "firstName": first_name
            }}

        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            print(f"DATABASE ERROR (create_user): {e}")
            return {"success": False, "message": "An internal database error occurred."}
        finally:
            if conn:
                conn.close()
    
    def get_documents_by_user(self, user_id):
        """Fetches list of documents uploaded by a specific user."""
        conn = None
        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()
            
            # Fetch id, file_name, and creation date
            cur.execute("SELECT id, file_name, uploaded_at FROM documents WHERE user_id = %s ORDER BY uploaded_at DESC, id DESC", (int(user_id),))
            rows = cur.fetchall()
            
            # Convert to list of dicts
            return [{"id": r[0], "title": r[1], "file_name": r[1], "date": str(r[2]) if r[2] else "Recent"} for r in rows]
        except Exception as e:
            print(f"DATABASE ERROR (get_documents_by_user): {e}")
            return []
        finally:
            if conn: conn.close()

    def get_summary_by_id(self, doc_id):
        """Fetches summary and keywords for a specific document ID."""
        conn = None
        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()

            # Get Summary
            cur.execute("SELECT summary_text FROM summaries WHERE document_id = %s", (doc_id,))
            row = cur.fetchone()
            if not row:
                return None
            summary_text = row[0]

            # Get Keywords
            cur.execute("SELECT keyword FROM keywords WHERE document_id = %s", (doc_id,))
            keywords = [r[0] for r in cur.fetchall()]

            return {
                "summary": summary_text,
                "keywords": keywords,
                "refined_summary": "" # Current DB schema doesn't store refined summary
            }
        except Exception as e:
            print(f"DATABASE ERROR (get_summary_by_id): {e}")
            return None
        finally:
            if conn: conn.close()

    def authenticate_user(self, username_or_email, password):
        """Authenticates a user by username/email and password."""
        conn = None
        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()

            cur.execute(
                "SELECT id, first_name, password_hash FROM users WHERE username = %s OR email = %s",
                (username_or_email, username_or_email)
            )
            user_record = cur.fetchone()

            cur.close()

            if user_record:
                user_id, first_name, password_hash = user_record
                if check_password_hash(password_hash, password):
                    return {"success": True, "user": {"id": user_id, "firstName": first_name}}

            return {"success": False, "message": "Invalid credentials"}

        except psycopg2.Error as e:
            print(f"DATABASE ERROR (authenticate_user): {e}")
            return {"success": False, "message": "An internal database error occurred"}
        finally:
            if conn:
                conn.close()

    def get_all_data(self):
        """Fetches and prints all records from the documents table."""
        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()
            
            # Ensure table exists before trying to select from it
            cur.execute("SELECT to_regclass('public.documents')")
            if cur.fetchone()[0] is None:
                print("\n⚠️ The 'documents' table does not exist. Nothing to show.")
                cur.close()
                conn.close()
                return

            cur.execute("SELECT id, user_id, file_name, uploaded_at FROM documents ORDER BY uploaded_at DESC")
            records = cur.fetchall()
            
            cur.close()
            conn.close()
            
            print(f"\n--- Found {len(records)} Record(s) in 'documents' ---")
            for record in records:
                print(f"\n  Doc ID: {record[0]}\n  User ID: {record[1]}\n  File: {record[2]}\n  Saved: {record[3]}")
            print("\n" + "="*50)

        except Exception as e:
            print(f"\n⚠️ DATABASE ERROR while fetching data: {e}")

    def inspect_latest_record(self):
        """Fetches and displays the most recently saved document, summary, and keywords."""
        try:
            conn = psycopg2.connect(**self.config)
            cur = conn.cursor()

            # 1. Get latest document
            cur.execute("""
                SELECT id, file_name, uploaded_at 
                FROM documents 
                ORDER BY uploaded_at DESC 
                LIMIT 1
            """)
            doc = cur.fetchone()
            
            if not doc:
                print("\n❌ No documents found in the database.")
                cur.close()
                conn.close()
                return

            doc_id, file_name, uploaded_at = doc
            print("\n" + "="*50)
            print(f"🔍 LATEST DATABASE ENTRY (ID: {doc_id})")
            print(f"   📂 File: {file_name}")
            print(f"   🕒 Time: {uploaded_at}")

            # 2. Get Summary
            cur.execute("SELECT summary_text FROM summaries WHERE document_id = %s", (doc_id,))
            summary_row = cur.fetchone()
            summary_text = summary_row[0] if summary_row else "⚠️ No summary found."
            preview = (summary_text[:150] + '...') if len(summary_text) > 150 else summary_text
            print(f"\n   📄 Summary Preview:\n   \"{preview}\"")

            # 3. Get Keywords
            cur.execute("SELECT keyword FROM keywords WHERE document_id = %s", (doc_id,))
            keywords = [row[0] for row in cur.fetchall()]
            print(f"\n   🔑 Keywords ({len(keywords)}):")
            print(f"   {', '.join(keywords)}")
            
            print("="*50 + "\n")

            cur.close()
            conn.close()

        except Exception as e:
            print(f"\n⚠️ DATABASE ERROR: {e}")