#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

class vertex { 
public:
	double x, y, z;
	double **Q;
	set<int> face_list;
	vertex(double _x = 0., double _y = 0., double _z = 0.): x(_x), y(_y), z(_z) {
		Q = new double*[4];
		for (int i = 0; i < 4; i++) {
			Q[i] = new double[4];
			Q[i][0] = Q[i][1] = Q[i][2] = Q[i][3] = 0;
		}
	}
	void computeQ();
};

class face {
public:
	int v1, v2, v3;
	double a, b, c, d;
	face(int _v1 = 0, int _v2 = 0, int _v3 = 0): v1(_v1), v2(_v2), v3(_v3) {}
	void computePara();
};

// QEM parameters
double simp_rate, dist_eps;
int face_num, vertex_num;

// For serperate
double min_x = DBL_MAX, min_y = DBL_MAX, min_z = DBL_MAX;
double max_x = -DBL_MAX, max_y = -DBL_MAX, max_z = -DBL_MAX;

vector<vertex> vertices;
vector<vector<vertex>> grid_vertices(8);

vector<face> faces;
vector<vector<face>> grid_faces(8);

vector<set<int>> grid_edges[8], grid_orig_edges[8];

int main(int argc, char *argv[]) {
    int nthreads = 8; // omp_get_max_threads()
    if (argc < 5) {
        puts("Usage: mesh_simp [input model] [output model] [simp rate] [dist threshold]");
        return 0;
    }
    
    freopen(argv[1], "r", stdin);
    freopen(argv[2], "w", stdout);
    simp_rate = atof(argv[3]);
    dist_eps = atof(argv[4]);
    ios::sync_with_stdio(false);

    omp_set_num_threads(nthreads);
    fprintf(stderr, "Running with %d threads\n", nthreads);

    char c;
    while ((c = getchar()) && (c != EOF)) {
        if (c == '#') {
            while ((c = getchar()) && (c != EOF) && (c != '\n'));
            continue;
        }
        if (c == 'v') {
            double x, y, z;
            scanf("%lf%lf%lf", &x, &y, &z);
            vertices.push_back(vertex(x, y, z));
            vertex_num++;
            // Update bounding box
            min_x = min(min_x, x);
            min_y = min(min_y, y);
            min_z = min(min_z, z);
            max_x = max(max_x, x);
            max_y = max(max_y, y);
            max_z = max(max_z, z);
        } else if (c == 'f') {
            int u, v, w;
            scanf("%d%d%d", &u, &v, &w);
            u--; v--; w--;
            faces.push_back(face(u, v, w));
            assert((u < vertex_num) && (v < vertex_num) && (w < vertex_num));
            vertices[u].face_list.insert(face_num);
            vertices[v].face_list.insert(face_num);
            vertices[w].face_list.insert(face_num);
            face_num++;
        }
    }
    
    // Calculate midpoints
    double mid_x = (min_x + max_x) / 2.0;
    double mid_y = (min_y + max_y) / 2.0;
    double mid_z = (min_z + max_z) / 2.0;

    // Assign vertices to regions
    for (int i = 0; i < vertex_num; i++) {
        double x = vertices[i].x;
        double y = vertices[i].y;
        double z = vertices[i].z;
        
        int index = 0;
        if (x > mid_x) index |= 1;
        if (y > mid_y) index |= 2;
        if (z > mid_z) index |= 4;
        
        grid_vertices[index].push_back(vertices[i]);
    }
    /*
    int total_count = 0;
    for (int i = 0; i < 8; i++) {
        int block_count = grid_vertices[i].size();
        total_count += block_count;
        fprintf(stderr, "Block %d: %d vertices\n", i, block_count);
    }
    fprintf(stderr, "Total vertices match: %d, Actual total: %d\n", total_count, vertex_num);
    */
    for (int i = 0; i < face_num; i++) {
        // 獲取該面的三個頂點索引
        int u = faces[i].v1, v = faces[i].v2, w = faces[i].v3;

        // 計算三個頂點的區塊索引
        double x1 = vertices[u].x, y1 = vertices[u].y, z1 = vertices[u].z;
        double x2 = vertices[v].x, y2 = vertices[v].y, z2 = vertices[v].z;
        double x3 = vertices[w].x, y3 = vertices[w].y, z3 = vertices[w].z;

        int index1 = 0, index2 = 0, index3 = 0;
        if (x1 > mid_x) index1 |= 1;
        if (y1 > mid_y) index1 |= 2;
        if (z1 > mid_z) index1 |= 4;

        if (x2 > mid_x) index2 |= 1;
        if (y2 > mid_y) index2 |= 2;
        if (z2 > mid_z) index2 |= 4;

        if (x3 > mid_x) index3 |= 1;
        if (y3 > mid_y) index3 |= 2;
        if (z3 > mid_z) index3 |= 4;

        // 確保三個頂點的區塊索引相同，才能將該面加入對應區塊
        if (index1 == index2 && index2 == index3) {
            grid_faces[index1].push_back(faces[i]);
        }
    }
    /*
    int total_count = 0;
    for (int i = 0; i < 8; i++) {
        int block_count = grid_faces[i].size();
        total_count += block_count;
        fprintf(stderr, "Block %d: %d faces\n", i, block_count);
    }
    fprintf(stderr, "Total faces match: %d, Actual total: %d\n", total_count, face_num);
    */
    
    // Initialize edges for each region
    for (int i = 0; i < 8; i++) {
        grid_edges[i].resize(vertex_num);
        grid_orig_edges[i].resize(vertex_num);
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < grid_faces[i].size(); j++) {
            int u = grid_faces[i][j].v1, v = grid_faces[i][j].v2, w = grid_faces[i][j].v3;

            // 確保三個頂點索引合法，且頂點不重複
            assert((u != v) && (v != w) && (w != u));

            // 計算三個頂點的區塊索引
            int index_u = (vertices[u].x > mid_x) | ((vertices[u].y > mid_y) << 1) | ((vertices[u].z > mid_z) << 2);
            int index_v = (vertices[v].x > mid_x) | ((vertices[v].y > mid_y) << 1) | ((vertices[v].z > mid_z) << 2);
            int index_w = (vertices[w].x > mid_x) | ((vertices[w].y > mid_y) << 1) | ((vertices[w].z > mid_z) << 2);

            // 如果三個頂點都屬於同一區塊，將邊加入對應的結構
            if (index_u == index_v && index_v == index_w) {
                int index = index_u;
                grid_edges[index][u].insert(v);
                grid_edges[index][u].insert(w);
                grid_edges[index][v].insert(u);
                grid_edges[index][v].insert(w);
                grid_edges[index][w].insert(u);
                grid_edges[index][w].insert(v);

                grid_orig_edges[index][u].insert(v);
                grid_orig_edges[index][u].insert(w);
                grid_orig_edges[index][v].insert(u);
                grid_orig_edges[index][v].insert(w);
                grid_orig_edges[index][w].insert(u);
                grid_orig_edges[index][w].insert(v);
            }
        }
    }
    pre = new int[vertex_num];
	for (int i = 0; i < vertex_num; i++) pre[i] = i;
	for (int i = 0; i < face_num; i++) faces[i].computePara();
	for (int i = 0; i < vertex_num; i++) vertices[i].computeQ();
	
    
    return 0;
}
