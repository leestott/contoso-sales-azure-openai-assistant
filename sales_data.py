import sqlite3
import pandas as pd
import json
from pydantic import BaseModel

DATA_BASE = "./database/contoso-sales.db"


class QueryResults(BaseModel):
    display_format: str = ""
    json_format: str = ""


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

    def __get_regions(self: "SalesData") -> list:
        """Return a list of unique regions in the database."""
        regions = self.conn.execute("SELECT DISTINCT region FROM sales_data;").fetchall()
        # convert list of tuples to list of strings
        return [region[0] for region in regions]


    def __get_product_types(self: "SalesData") -> list:
        """Return a list of unique product types in the database."""
        product_types = self.conn.execute("SELECT DISTINCT product_type FROM sales_data;").fetchall()
        # convert list of tuples to list of strings
        return [product_type[0] for product_type in product_types]

    def __get_product_categories(self: "SalesData")-> list:
        """Return a list of unique product categories in the database."""
        product_categories = self.conn.execute("SELECT DISTINCT main_category FROM sales_data;").fetchall()
        # convert list of tuples to list of strings
        return [product_category[0] for product_category in product_categories]


    def __get_reporting_years(self: "SalesData") -> list:
        """Return a list of unique reporting years in the database."""
        reporting_years = self.conn.execute("SELECT DISTINCT year FROM sales_data ORDER BY year;").fetchall()
        # convert list of tuples to list of strings
        return [str(reporting_year[0]) for reporting_year in reporting_years]

    def get_database_info(self: "SalesData") -> str:
        """Return a string containing the database schema information and common query fields."""
        table_dicts = []
        for table_name in self.__get_table_names():
            columns_names = self.__get_column_info(table_name)
            table_dicts.append({"table_name": table_name, "column_names": columns_names})

        database_info = "\n".join(
            [
                f"Table {table['table_name']} Schema: Columns: {', '.join(table['column_names'])}"
                for table in table_dicts
            ]
        )
        regions = self.__get_regions()
        product_types = self.__get_product_types()
        product_categories = self.__get_product_categories()
        reporting_years = self.__get_reporting_years()

        database_info += f"\nRegions: {', '.join(regions)}"
        database_info += f"\nProduct Types: {', '.join(product_types)}"
        database_info += f"\nProduct Categories: {', '.join(product_categories)}"
        database_info += f"\nReporting Years: {', '.join(reporting_years)}"
        database_info += "\n\n"

        return database_info

    def ask_database(self: "SalesData", query: str) -> QueryResults:
        """Function to query SQLite database with a provided SQL query."""

        data_results = QueryResults()

        try:
            data = pd.read_sql_query(query, self.conn)
            if data.empty:
                data_results.display_format = "The query returned no results. Try a different query."
                data_results.json_format = ""
                return data_results
            data_results.display_format = data.to_string()
            table = json.dumps(data.to_json(index=False, orient="split"))
            data_results.json_format = table

        except Exception as e:
            data_results.display_format = f"query failed with error: {e}"
            data_results.json_format = "{" + f'"error": "{e}", "query":"{query}"' + "}"
        return data_results
