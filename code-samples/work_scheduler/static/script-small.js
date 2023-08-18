let projects = [];
let tasks = [];

let markdownEditor;  // set later once to-dos are loaded

function createProject(project) {
    let projectId = projects.length; // new index at the end
    let today = new Date();
    let assigned = new Date(project.assigned);
    let due = new Date(project.due);
    let finishBy = new Date(project.finishBy);

    // Add 1 so that these include working on the due date / finish by date
    let dueDateDifference = Math.round(1 + (due.getTime() - today.getTime()) / (1000 * 3600 * 24));
    let finishByDifference = Math.round(1 + (finishBy.getTime() - today.getTime()) / (1000 * 3600 * 24));

    // TODO: Fix this, later...
    let daysElapsed = Math.round((today.getTime() - assigned.getTime()) / (1000 * 3500 * 24));
    let daysTotal = Math.round(1 + (finishBy.getTime() - assigned.getTime()) / (1000 * 3600 * 24));

    console.log(daysElapsed);
    console.log(daysTotal);

    let onTimeIdeal = 100 * (daysElapsed / daysTotal);
    if (onTimeIdeal < 0) {
        onTimeIdeal = 0; // set to zero so that future assigned date progress bar is empty
    }

    let projectHTML = `<li class="list-group-item" id="project-${projectId}">
    <button class="btn btn-dark badge mr-2" onclick="updateParameter(${projectId}, 'project', 'name')">🖉</button><b>Name: </b>${project.name} <br />
    <button class="btn btn-dark badge mr-2" onclick="updateParameter(${projectId}, 'project', 'assigned')">🖉</button><b>Assigned: </b>${project.assigned}<br />
    <button class="btn btn-dark badge mr-2" onclick="updateParameter(${projectId}, 'project', 'due')">🖉</button><b>Due date: </b>${project.due} (${dueDateDifference} days)<br />
    <button class="btn btn-dark badge mr-2" onclick="updateParameter(${projectId}, 'project', 'finishBy')">🖉</button><b>Finish by date: </b>${project.finishBy} (${finishByDifference} days)<br />
    <div class="progress bg-dark" style="height: 24px">
        <div class="progress-bar bg-primary" style="width: ${project.progress}%; height: 24px">
            Current work progress (${project.progress}%)
        </div>
    </div>
    <div class="progress mt-2 bg-dark" style="height: 24px">
        <div class="progress-bar bg-success" style="width: ${onTimeIdeal}%; height: 24px">
            Needed progress to be on track (${Math.round(onTimeIdeal)}%)
        </div>
    </div>
    <button class="btn btn-primary mt-2" onclick="addProgress(${projectId})">Add progress</button>
    <button class="btn btn-warning mt-2" onclick="updateItem(${projectId}, 'project')">Update info</button>
    <button class="btn btn-danger mt-2" onclick="deleteItem(${projectId}, 'project')">Delete</button>
    </li>`;

    $("#projects").html($("#projects").html() + projectHTML);
    projects.push(project);
}

function createTask(task) {
    let taskId = tasks.length;
    let today = new Date();
    let due = new Date(task.due);
    let finishBy = new Date(task.finishBy);

    // Add 1 so that these include working on the due date / finish by date
    let dueDateDifference = Math.round(1 + (due.getTime() - today.getTime()) / (1000 * 3600 * 24));
    let finishByDifference = Math.round(1 + (finishBy.getTime() - today.getTime()) / (1000 * 3600 * 24));

    // Use this to conditionally set background to in progress colors
    let inProgressLabel = (task.inProgress) ? "bg-in-progress text-light" : "bg-not-started text-light";

    let taskHTML = `<li class="list-group-item ${inProgressLabel}" id="task-${taskId}">
    <b onclick="updateParameter(${taskId}, 'task', 'name')">${task.name}</b>, 
    due in <b onclick="updateParameter(${taskId}, 'task', 'due')">${dueDateDifference} </b>days, 
    finish in <b onclick="updateParameter(${taskId}, 'task', 'finishBy')">${finishByDifference} </b> days<br />
    <div class="btn-group">
    <button class="btn bg-not-started text-light mt-2" onclick="taskNS(${taskId})">NS</button>
    <button class="btn bg-in-progress text-light mt-2" onclick="taskIP(${taskId})">IP</button>
    <button class="btn btn-success text-light mt-2" onclick="taskDone(${taskId})">Done</button>
    
    </div>
    <button class="btn btn-light text-dark mt-2" onclick="updateItem(${taskId}, 'task')">Update info</button>
    <button class="btn btn-light text-dark mt-2" onclick="deleteItem(${taskId}, 'task')">Delete</button>
    </li>`;

    $("#projects").html($("#projects").html() + taskHTML);
    tasks.push(task);
}

function loadTasks() {
    $.get("/data/tasks.json", (data) => {
        data = JSON.parse(data);
        data.sort((a, b) => {
            return (a.finishBy < b.finishBy) ? -1 : 1;
        });
        for (let i = 0; i < data.length; i++) {
            if (data[i].type == "project") {
                createProject(data[i]);
            } else if (data[i].type == "task") {
                createTask(data[i]);
            }
        }
    });
}

function loadTodos() {
    $.get("/data/todo.md", (data) => {
        $("#todos").val(data);
        markdownEditor = new SimpleMDE({
            spellChecker: false, renderingConfig: {
                codeSyntaxHighlighting: true
            }
        });
        markdownEditor.codemirror.on("change", () => {
            $.post("/data/todo.md", { 'data': markdownEditor.value() });
        });
    });
}

// depreciated since switch to markdown editor
// $("#todos").change(() => {
//     $.post("/data/todo.txt", { 'data': $("#todos").val() });
// });

function sortItems() {
    projects.sort((a, b) => {
        return (a.finishBy < b.finishBy) ? -1 : 1;
    });

    tasks.sort((a, b) => {
        return (a.finishBy < b.finishBy) ? -1 : 1;
    });
}

function addTask() {
    let taskName = prompt("Enter name of task: ");
    if (taskName == null) return;

    let dueDateString = prompt("Enter due date (YYYY-MM-DD): ");
    if (dueDateString == null) return;

    let finishByString = prompt("Enter finish by date (leave blank if same as due date): ");
    if (finishByString == null) return;
    else if (finishByString == '') finishByString = dueDateString;

    let newTask = {
        'name': taskName,
        'type': 'task',
        'due': dueDateString,
        'finishBy': finishByString,
        'inProgress': false
    };

    createTask(newTask);
    updateFile();
    refreshDisplays();
}

function addProject() {
    let projectName = prompt("Enter name of project: ");
    if (projectName == null) return;

    let dueDateString = prompt("Enter due date (YYYY-MM-DD): ");
    if (dueDateString == null) return;

    let finishByString = prompt("Enter finish by date (leave blank if same as due date): ");
    if (finishByString == null) return;
    else if (finishByString == '') finishByString = dueDateString;

    let assignedDate = prompt("Enter date of assignment (YYYY-MM-DD): ");
    if (assignedDate == null) return;

    let newProject = {
        'name': projectName,
        'type': 'project',
        'assigned': assignedDate,
        'due': dueDateString,
        'finishBy': finishByString,
        'progress': 0
    };

    createProject(newProject);
    updateFile();
    refreshDisplays();
}

function updateItem(id, type) {
    let item;
    if (type == 'project') {
        item = projects[id];
    } else if (type == 'task') {
        item = tasks[id];
    }

    let newDesc = prompt("Enter new description (leave blank to skip):");
    if (newDesc != null && newDesc != "") {
        item.name = newDesc;
    }

    let newDue = prompt("Enter new due date (leave blank to skip):");
    if (newDue != null && newDue != "") {
        item.due = newDue;
    }

    let newFinishBy = prompt("Enter new finish by date (leave blank to skip):");
    if (newFinishBy != null && newFinishBy != "") {
        item.finishBy = newFinishBy;
    }

    updateFile();
    refreshDisplays();
}

// Example invocation: updateParameter(5, 'project', 'name'|'due'|'finishBy')
function updateParameter(id, type, field) {
    let item;
    if (type == 'project') {
        item = projects[id];
    } else if (type == 'task') {
        item = tasks[id];
    }

    let currentVal = item[field];
    let newVal = prompt(`Enter new value for field "${field}": `, currentVal);
    console.log(newVal);
    if (newVal == "" || newVal == null) {
        return;
    }

    item[field] = newVal;

    updateFile();
    refreshDisplays();
}

function addProgress(id) {
    let newAmount = prompt(`Enter new progress: `, projects[id].progress);
    if (newAmount == null) return;
    newAmount = parseInt(newAmount);

    projects[id].progress = newAmount;

    if (newAmount == 100) {
        deleteItem(id, 'project'); // this will prompt to confirm
    }

    updateFile();
    refreshDisplays();
}

function deleteItem(id, type) {
    if (!confirm("Are you sure you want to delete this item?")) return;
    if (type == "project") projects.splice(id, 1); // remove one element starting at id
    else if (type == "task") delete tasks.splice(id, 1);

    updateFile();
    refreshDisplays();
}

function taskNS(id) {
    tasks[id].inProgress = false;

    updateFile();
    refreshDisplays();
}

function taskIP(id) {
    tasks[id].inProgress = true;

    updateFile();
    refreshDisplays();
}

function taskDone(id) {
    if (!confirm("Are you sure you are done with this task?")) return;
    if (confirm("Repeat this task?")) {
        let repeatString = prompt("Enter number of days until this task repeats: ");
        if (repeatString == null) return;

        let numDays = parseInt(repeatString);
        let due = new Date(tasks[id].due);
        due.setDate(due.getDate() + numDays);
        tasks[id].due = due.toISOString().substring(0, 10);

        let finishBy = new Date(tasks[id].finishBy);
        finishBy.setDate(finishBy.getDate() + numDays);
        tasks[id].finishBy = finishBy.toISOString().substring(0, 10);

        // let dueDateString = prompt("Enter due date (YYYY-MM-DD): ");
        // if (dueDateString == null) return;

        // let finishByString = prompt("Enter finish by date (leave blank if same as due date): ");
        // if (finishByString == null) return;
        // else if (finishByString == '') finishByString = dueDateString;

        // tasks[id].due = dueDateString;
        // tasks[id].finishBy = finishByString;
        tasks[id].inProgress = false;

        updateFile();
        refreshDisplays();
    } else {
        tasks.splice(id, 1);

        updateFile();
        refreshDisplays();
    }
}


function refreshDisplays() {
    location.reload();
    return;
    // TODO: get this to work without refreshing

    $("#projects").html("");
    $("#tasks").html("");

    let numProjects = projects.length;
    let numTasks = tasks.length;

    for (let i = 0; i < numProjects; i++) {
        createProject(projects[0]);  // this will duplicate and put at end of projects array
        delete projects[0]; // delete what we just worked on
    }

    for (let i = 0; i < numTasks; i++) {
        createTask(tasks[0]);
        delete tasks[0];
    }
}

function updateFile() {
    sortItems(); // sort before saving

    let items = [];
    for (let i = 0; i < projects.length; i++) {
        items.push(projects[i]);
    }

    for (let i = 0; i < tasks.length; i++) {
        items.push(tasks[i]);
    }

    $.post("/data/tasks.json", { 'data': JSON.stringify(items) });
}

$(document).ready(() => {
    loadTodos();
    loadTasks();
});