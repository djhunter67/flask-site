import mysql.connector


def main():
    try:
        connection = mysql.connector.connect(host='192.168.110.24',
                                             database='test',
                                             user='djhunter67',
                                             password='DaniellE08!!**!!')

        cursor = connection.cursor()

        print(f"Contents: {cursor.fetchall()}")

    except mysql.connector.Error as error:
        print(f"MySQL connection has failed: {error}")

    finally:
        if connection.is_connected():
            connection.close()
            cursor.close()
            print("MySQL connection closed")


if __name__ == "__main__":
    main()
