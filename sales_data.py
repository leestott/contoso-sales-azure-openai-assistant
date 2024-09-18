import sqlite3
import pandas as pd

DATA_BASE = "./database/contoso-sales.db"


class SalesData:
    def __init__(self: "SalesData") -> None:
        self.conn = sqlite3.connect(DATA_BASE)

    def __get_table_names(self: "SalesData") -> list:
        """Return a list of table names."""
        table_names = []
        tables = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for table in tables.fetchall():
            if table[0] != "sqlite_sequence":
                table_names.append(table[0])
        return table_names

    def __get_column_info(self: "SalesData", table_name: str) -> list:
        """Return a list of tuples containing column names and their types."""
        column_info = []
        columns = self.conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
        for col in columns:
            column_info.append(f"{col[1]}: {col[2]}")  # col[1] is the column name, col[2] is the column type
        return column_info

    def get_database_info(self: "SalesData") -> list:
        """Return a list of dicts containing the table name and columns for each table in the database."""
        table_dicts = []
        for table_name in self.__get_table_names():
            columns_names = self.__get_column_info(table_name)
            table_dicts.append({"table_name": table_name, "column_names": columns_names})
        return table_dicts

    def ask_database(self: "SalesData", query: str) -> list:
        """Function to query SQLite database with a provided SQL query."""
        results = []

        try:
            data = pd.read_sql_query(query, self.conn)
            if data.empty:
                results.append(f"Query: {query}\n\n")
                results.append("No results found.")
                return results
            results.append(f"Query: {query}\n\n")
            table = data.to_string(index=False)
            results.append(table)

        except Exception as e:
            results.append(f"Query: {query}")
            results.append(f"query failed with error: {e}")
        return results
