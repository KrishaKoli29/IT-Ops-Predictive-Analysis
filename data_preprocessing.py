import pandas as pd # type: ignore

def prepare_ticket_data(file_path):
    print(f"Loading data from {file_path}:\n")

    df = pd.read_csv(file_path)

    df['Date_Created'] = pd.to_datetime(df['Date_Created'])   #raw->pandas datetime
    df['Date_Only'] = df['Date_Created'].dt.date #removing hrs/mins

    daily_ticket_volume = df.groupby('Date_Only')['Ticket_ID'].count().reset_index() #chronological
    daily_ticket_volume.columns = ['Date', 'Total_Tickets'] #date_only and ids
    
    return daily_ticket_volume

if __name__ == "__main__":
    final_data = prepare_ticket_data('helpdesk_tickets.csv')
    
    print("--- Preprocessed Daily Ticket Volume ---")
    print(final_data)