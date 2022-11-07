import plotly.express as px

with open("data.txt","r") as f:
	a = f.read().split(',')


a = list(map(lambda x : float(x), a))
z = [(i/200 * 100) for i in range(1,201)]

fig = px.line(x = z, y = a)
fig.update_layout(
    title = "Accuracy of model in relation of the data fed to B Gradient Descent",
	yaxis_title = "COST",
	xaxis_title = "% of Dataset used"
)

fig.show()