import pycurl
from io import BytesIO
import json
import csv



access_token = ["****"]
access_token_counter = 0

def closedIssue(repo_full_name):
    pageCounter = 1

    file = open(file="spark.csv", mode="w")
    file_writer = csv.writer(file)
    file_writer.writerow(["issue_id", "created_date", "closed_date"])

    while True:
        global access_token_counter
        url = f"https://api.github.com/repos/{repo_full_name}/issues?state=closed&per_page=100&page={pageCounter}"
        output = BytesIO()
        request = pycurl.Curl()
        request.setopt(pycurl.HTTPHEADER, [f'Authorization: token {access_token[access_token_counter % 4]}'])
        request.setopt(request.URL, url)
        request.setopt(request.WRITEDATA, output)
        request.perform()
        get_body = output.getvalue().decode()
        body = json.loads(get_body)

        if body == []:
            break
        print(f"Page number {pageCounter} is processing")
        for issue in body:
            issue_id = issue['id']
            create_date = issue["created_at"]
            closed_date = issue['closed_at']
            file_writer.writerow([issue_id, create_date, closed_date])


        access_token_counter += 1
        pageCounter += 1

    file.close()

if __name__ == '__main__':
    # closedIssue("openai/gym")
    # closedIssue("tensorforce/tensorforce")
    # closedIssue("google/dopamine")
    closedIssue("apache/spark")