import slackweb

slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
for i in range(10):
    slack.notify(text=f"hello {i}")