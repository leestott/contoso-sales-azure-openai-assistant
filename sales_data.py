import sqlite3
import pandas as pd
import json

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
    
    def __get_regions(self: "SalesData"):
        """Return a list of unique regions in the database."""
        regions = self.conn.execute("SELECT DISTINCT region FROM sales_data;").fetchall()
        # convert list of tuples to list of strings
        regions = [region[0] for region in regions]
        return regions
    
    def __get_product_types(self: "SalesData"):
        """Return a list of unique product types in the database."""
        product_types = self.conn.execute("SELECT DISTINCT product_type FROM sales_data;").fetchall()
        # convert list of tuples to list of strings
        product_types = [product_type[0] for product_type in product_types]
        return product_types
    
    def __get_product_categories(self: "SalesData"):
        """Return a list of unique product categories in the database."""
        product_categories = self.conn.execute("SELECT DISTINCT main_category FROM sales_data;").fetchall()
        # convert list of tuples to list of strings
        product_categories = [product_category[0] for product_category in product_categories]
        return product_categories

    def get_database_info(self: "SalesData") -> str:
        """Return a list of dicts containing the table name and columns for each table in the database."""
        table_dicts = []
        for table_name in self.__get_table_names():
            columns_names = self.__get_column_info(table_name)
            table_dicts.append({"table_name": table_name, "column_names": columns_names})
        # return table_dicts
        database_info = "\n".join(
            [f"Table {table['table_name']} Schema: Columns: {', '.join(table['column_names'])}" for table in table_dicts]
        )
        regions = self.__get_regions()
        product_types = self.__get_product_types()
        product_categories = self.__get_product_categories()

        database_info += f"\nRegions: {', '.join(regions)}"
        database_info += f"\nProduct Types: {', '.join(product_types)}"
        database_info += f"\nProduct Categories: {', '.join(product_categories)}"
        database_info += "\n\n"

        return database_info

    def ask_database(self: "SalesData", query: str) -> list:
        """Function to query SQLite database with a provided SQL query."""
        results = []

        try:
            data = pd.read_sql_query(query, self.conn)
            if data.empty:
                results.append(f"Query: {query}\n\n")
                results.append("The query returned no results. Try a different query. ")
                return results
            results.append(f"Query: {query}\n\n")
            table = json.dumps(data.to_json(index=False, orient='split'))
            # results.append(f"Query JSON Data: {table}")
            results.append(table)

        except Exception as e:
            results.append(f"Query: {query}")
            results.append(f"query failed with error: {e}")
        return results
